import argparse
import torch.nn as nn
import numpy as np
from pathlib import Path
import torch.nn.utils
import torch.optim as optim
import yaml
import time
from copy import deepcopy
from YOLOLITE.loss import ComputeLoss
from torch.cuda import amp
import torch.optim.lr_scheduler as lr_scheduler
from YOLOLITE.test import test
from YOLOLITE.model.yololite import Model
from YOLOLITE.model.common import ModelEMA
from YOLOLITE.data import create_dataloader
from YOLOLITE.utils import make_logger, intersect_dicts, one_cycle, check_img_size, check_anchors, labels_to_class_weights, fitness

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', action='store_false', help='Whether to load the pretrained model')
parser.add_argument('--weights', type=str, default='./YOLOLITE/save_nowarm/lastest.pth', help='initial weights path')
parser.add_argument('--cfg', type=str, default='./YOLOLITE/model/v5Lite-s.yaml', help='model.yaml path')
parser.add_argument('--data', type=str, default='./YOLOLITE/data/voc.yaml', help='data.yaml path')
parser.add_argument('--hyp', type=str, default='./YOLOLITE/data/hyp.finetune.yaml', help='hyperparameters path')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--test_freq', type=int, default=5, help='the frequence for evaluate model in test dataset during training phase')
parser.add_argument('--ema', action='store_false', help='Model Exponential Moving Average')
parser.add_argument('--test_half', action='store_false', help='Test with half precision')
parser.add_argument('--batch_size', type=int, default=8, help='total batch size for all GPUs')
parser.add_argument('--img_size', nargs='+', type=int, default=[512, 512], help='[train, test] image sizes')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
parser.add_argument('--image_weights', action='store_true', help='use weighted image selection for training')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
parser.add_argument('--linear-lr', action='store_true', help='linear LR')
parser.add_argument('--quad', action='store_true', help='quad dataloader')
parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
parser.add_argument('--save_dir', type=str, default="./YOLOLITE/save_nowarm_steplr_2005", help='the dir to save output during traing/testing phase')
parser.add_argument('--combine', action='store_true', help='is trained combine sr and od')


def train(hyp, data_cfg, cfg, opt, logger=None, test_step=1):
    opt.save_dir = Path(opt.save_dir)
    with open(opt.save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(opt.save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    with open(opt.save_dir / 'cfg.yaml', 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    with open(opt.save_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_cfg, f, sort_keys=False)

    num_cls = data_cfg['nc']
    names = data_cfg['names']
    assert len(names) == num_cls, '定义的类别数量和找到的类名的总数不一致'  # check

    if opt.pretrained:  # 加载coco 预训练模型
        ckpt = torch.load(opt.weights)  # load checkpoint
        model = Model(cfg, ch=3, nc=num_cls).cuda()  # create todo 网络配置文件, 输入通道, 类别数, None  构建网络
        state_dict = ckpt['model'].float().state_dict() # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=['anchor'])  # 加载除了detect 参数 和 anchor 参数之外的其他参数
        model.load_state_dict(state_dict, strict=False)
        logger.info('loaded {}/{} parameter items from {}'.format(len(state_dict), len(model.state_dict()), opt.weights))  # report
    else:
        model = Model(cfg, ch=3, nc=num_cls, anchors=hyp.get('anchors')).cuda()  # create

    # 不同参数设置不同的 weight_decay
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups   todo other, weight, bias
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay for bn.weight
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: {} .bias, {} conv.weight, {} other'.format(len(pg2), len(pg1), len(pg0)))

    if opt.linear_lr:
        lf = lambda x: (1 - x / (opt.epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], opt.epochs)  # cosine 1->hyp['lrf']

    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.5)

    if opt.ema:
        modelema = ModelEMA(model)

    # Resume
    start_epoch = 0

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # detect layer 的最大步长
    nl = model.model[-1].nl  # detect layer 有 nl 个尺度
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Trainloader
    dataloader, dataset = create_dataloader(data_cfg['train'], imgsz, opt.batch_size, gs, hyp=hyp, augment=True, rect=opt.rect,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix='train', combine=opt.combine)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches for training data
    assert mlc < num_cls, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, num_cls, opt.data, num_cls - 1)

    testloader = create_dataloader(data_cfg['val'], imgsz_test, opt.batch_size * 2, gs, hyp=hyp, rect=True, pad=0.5, prefix='val', combine=opt.combine)[0]
    if not opt.resume:
        check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= num_cls / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = num_cls  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, num_cls).cuda() * num_cls  # attach class weights
    model.names = names

    # Start training
    scaler = amp.GradScaler()
    # maps = np.zeros(num_cls)  # mAP per class
    # results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move

    compute_loss = ComputeLoss(model)  # init loss class
    logger.info('Image sizes {} train, {} test,  Using {} dataloader workers, Logging results to {}, Starting training for {} epochs...'
                .format(imgsz, imgsz_test, dataloader.num_workers, opt.save_dir, opt.epochs))
    t = time.time()
    for epoch in range(start_epoch, opt.epochs):  # epoch ------------------------------------------------------------------
        model.train()
        mloss = torch.zeros(4, device='cuda')  # mean losses
        for i, (imgs, targets, paths, _) in enumerate(dataloader):  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)  todo 迭代次数
            imgs = imgs.cuda().float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # # Warmup
            # if ni <= 1000:
            #     for j, x in enumerate(optimizer.param_groups):
            #         # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            #         x['lr'] = np.interp(ni, [0, 1000], [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])  # 0.05 / 0.0 ---> 0.0032*lf(epoch)
            #         x['momentum'] = np.interp(ni, [0, 1000], [hyp['warmup_momentum'], hyp['momentum']])  # 0.5 : 0.843

            # Forward
            with amp.autocast():
                pred = model(imgs)  # forward  # [torch.Size([64, 3, 52, 52, 25]), torch.Size([64, 3, 26, 26, 25]), torch.Size([64, 3, 13, 13, 25])]
                loss, loss_items = compute_loss(pred, targets.cuda())  # loss scaled by batch_size todo # loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

            # Backward && Optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if opt.ema:
                 modelema.update(model)

            #  记录一下损失
            mloss = (mloss * i + loss_items) / (i + 1)  # 当前 epoch 内已迭代 过的 batch 的平均损失
            logger.info('epoch {:>4}/{} | iter:{:>4}/{} | boxloss {} | objloss {} | clsloss {} | totalloss {} | labels {} | lr {} | time {}s'
                        .format(epoch + 1, opt.epochs, i + 1, nb, '%.6f' % mloss[0], '%.6f' % mloss[1], '%.6f' % mloss[2], '%.6f' % mloss[3],
                                targets.shape[0], scheduler.get_last_lr()[0], '%.6f' % (time.time() - t)))
            t = time.time()
        # Scheduler
        # lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()
        if opt.ema:
            modelema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])

        is_final_epoch = epoch + 1 == opt.epochs
        # Calculate mAP
        if (epoch + 1) % test_step == 0 or is_final_epoch:
            results, maps, times = test(data_cfg, batch_size=opt.batch_size * 2, imgsz=imgsz_test, model=modelema.ema if opt.ema else model,
                                        dataloader=testloader, save_dir=opt.save_dir, verbose=num_cls < 50 and is_final_epoch,
                                        plots=is_final_epoch, compute_loss=compute_loss, half = opt.test_half, logger=logger)

        ckpt = {'epoch': epoch + 1,
                'model': deepcopy(model).half(),
                'ema': deepcopy(modelema.ema).half(),
                'updates': modelema.updates,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
        torch.save(ckpt, opt.save_dir / 'lastest.pth')
        print('\n')


def run():
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    opt = parser.parse_args()
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps  todo 训练的超参数
    with open(opt.data) as f:
        data_cfg = yaml.load(f, Loader=yaml.SafeLoader)  # data dict todo 数据相关参数
    with open(opt.cfg) as f:
        model_cfg = yaml.load(f, Loader=yaml.SafeLoader)  # data dict todo 模型相关参数
    logger = make_logger(log_file=opt.save_dir)
    test_step = opt.epochs // opt.test_freq
    # test_step = 1
    train(hyp, data_cfg, model_cfg, opt, logger, test_step)
