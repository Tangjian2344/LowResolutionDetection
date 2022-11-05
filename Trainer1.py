import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from pathlib import Path
import torch.nn.utils
import torch.optim as optim
import yaml
import time
from collections import OrderedDict
from analyze import analyze_data
from copy import deepcopy
from YOLOLITE.loss import ComputeLoss
from torch.cuda import amp
import torch.optim.lr_scheduler as lr_scheduler
from YOLOLITE.test import test
from YOLOLITE.model.yololite import Model
from YOLOLITE.model.common import ModelEMA
from YOLOLITE.data import create_dataloader
from YOLOLITE.utils import make_logger, intersect_dicts, one_cycle, check_img_size, check_anchors, labels_to_class_weights, fitness, my_one_cycle
import cv2
from LROD.model import make_model


def train(hyp, data_cfg, cfg, opt, config_sr, logger=None, writer=None):
    opt.save_dir = Path(opt.save_dir)
    # 保存本次训练的相关参数
    with open(opt.save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(opt.save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    with open(opt.save_dir / 'cfg.yaml', 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    with open(opt.save_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_cfg, f, sort_keys=False)
    with open(opt.save_dir / 'sr.yaml', 'w') as f:
        yaml.dump(config_sr, f, sort_keys=False)
    del f
    num_cls = data_cfg['nc']
    names = data_cfg['names']
    assert len(names) == num_cls, '定义的类别数量和找到的类名的总数不一致'  # check

    if opt.pretrained:  # 加载coco 预训练模型  和 Div2k 预训练超分模型
        # 检测模型
        ckpt = torch.load(opt.weight_od) # load checkpoint
        model = Model(cfg, ch=3, nc=num_cls).cuda()  # create todo 网络配置文件, 输入通道, 类别数, None  构建网络
        if isinstance(ckpt['model'], OrderedDict):
            state_dict = ckpt['model']
        else:
            state_dict = ckpt['model'].float().state_dict()  # to FP32

        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=['anchor'])  # 加载除了detect 参数 和 anchor 参数之外的其他参数
        model.load_state_dict(state_dict, strict=False)
        logger.info('loaded {}/{} parameter items from {}'.format(len(state_dict), len(model.state_dict()), opt.weight_od))  # report

        # 超分模型
        model_sr = make_model(config_sr['Model'], opt).cuda()
        state_dict = torch.load(opt.weight_sr).float().state_dict()
        model_sr.load_state_dict(state_dict)
        del state_dict
    else:
        model = Model(cfg, ch=3, nc=num_cls, anchors=hyp.get('anchors')).cuda()  # create
        model_sr = make_model(config_sr['Model'], opt).cuda()
    del cfg

    if opt.ema:
        modelema = ModelEMA(model)

    # 检测网络优化器 weight_decay
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups   todo other, weight, bias
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay for bn.weight
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

    optimizer.add_param_group({'params': model_sr.parameters()})
    logger.info('Optimizer groups: {} .bias, {} conv.weight, {} other'.format(len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2, k, v

    # 检测网络学习率
    test_step = max(opt.epochs[1] // opt.test_freq, 1)

    # lf = one_cycle(1, hyp['lrf'], opt.epochs[1] + opt.epochs[0])  # cosine 1->hyp['lrf']
    lf = my_one_cycle(1, hyp['lrf'], opt.epochs[0], opt.epochs[1])  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_epoch = 0
    if opt.pass1:
        ckpt = torch.load(opt.weight_sr_fined)
        start_epoch = ckpt['epoch']
        logger.info('It will used the finetuned sr weight from {} to skip traing phase1 and traing continue from {} epoch'.format(opt.weight_sr_fined, start_epoch + 1))
        model_sr.load_state_dict(ckpt['model_sr'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

        del ckpt

    # Resume
    if opt.resume:
        ckpt = torch.load(opt.checkpoint)
        start_epoch = ckpt['epoch']
        logger.info('It will load checkpoint from {} and traing continue from {} epoch'.format(opt.checkpoint, start_epoch + 1))
        model_sr.load_state_dict(ckpt['model_sr'])
        model.load_state_dict(ckpt['model'].float().state_dict())
        if opt.ema:
            modelema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            modelema.updates = ckpt['updates']
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

        logger.info('Skiped the traing sr along phase, now traing from epoch: {}'.format(start_epoch + 1))
        del ckpt

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

    testloader = create_dataloader(data_cfg['val'], imgsz_test, opt.batch_size, gs, hyp=hyp, rect=True, pad=0.5, prefix='val', combine=opt.combine)[0]
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
    del gs, mlc, num_cls, nl

    # 统计训练集中的数据
    # analyze_data(dataloader, 'train')
    # analyze_data(testloader, 'test')

    # Start training
    scaler = amp.GradScaler()
    loss_sr = nn.L1Loss()
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info('Image sizes {} train, {} test,  Using {} workers, Logging to {}, training sr {} epochs along , combine traing {}'
                .format(imgsz, imgsz_test, dataloader.num_workers, opt.save_dir, opt.epochs[0], opt.epochs[1]))

    if opt.pass1:
        logger.info('load the finetuned sr net and the det test result in HR / LR(supered) images as follows:')
        test(data_cfg, batch_size=opt.batch_size, imgsz=imgsz_test, model=modelema.ema if opt.ema else model,
             model_sr=None, dataloader=testloader, save_dir=opt.save_dir, verbose=False,
             plots=False, compute_loss=compute_loss, half=opt.test_half, logger=logger)
        results, maps, times = test(data_cfg, batch_size=opt.batch_size, imgsz=imgsz_test, model=modelema.ema if opt.ema else model,
                                    model_sr=model_sr, dataloader=testloader, save_dir=opt.save_dir, verbose=False,
                                    plots=False, compute_loss=compute_loss, half=opt.test_half, logger=logger)
        writer.add_scalars('det_eval', {'Precision': results[0], 'Recall': results[1], 'mAP@.5': results[2], 'mAP@.5:.95': results[3]}, start_epoch)

    if opt.use_lr:
        pl = (imgsz - imgsz // 4) // 2
        pad_lr = torch.nn.ConstantPad2d(pl, 144.)

    t = time.time()

    for epoch in range(start_epoch, opt.epochs[0] + opt.epochs[1]):  # epoch ------------------------------------------------------------------
        model.train()
        model_sr.train()
        mloss = torch.zeros(4, device='cuda')  # mean losses
        for i, (imgs, lr_imgs, targets, paths, _) in enumerate(dataloader):
            # prepare data
            imgs = imgs.cuda().float()  # uint8 to float32, 0-255 to 0.0-1.0    / 255.
            lr_imgs = lr_imgs.cuda().float()
            targets = targets.cuda()
            optimizer.zero_grad()

            # Forward & backword sr net
            loss0 = 0
            if not opt.only_det:
                with amp.autocast():
                    sr_imgs = model_sr(lr_imgs)
                    loss0 = loss_sr(sr_imgs, imgs)
                if opt.alpha:
                    if epoch >= opt.epochs[0]:
                        scaler.scale(loss0 * opt.alpha).backward(retain_graph=True)
                    else:
                        scaler.scale(loss0).backward()

            if epoch >= opt.epochs[0]:
                with amp.autocast():
                    if opt.only_det:
                        sr_imgs = imgs
                    elif opt.use_hr:
                        tn, tb = targets.shape[0], targets[-1][0].item() + 1
                        sr_imgs = torch.cat([sr_imgs, imgs], dim=0)
                        targets = torch.cat([targets, targets + torch.tensor([tb, 0., 0., 0., 0., 0.], device=targets.device)])

                    pred = model(sr_imgs / 255.)  # forward  # [torch.Size([64, 3, 52, 52, 25]), torch.Size([64, 3, 26, 26, 25]), torch.Size([64, 3, 13, 13, 25])]
                    loss, loss_items = compute_loss(pred, targets.cuda())  # loss scaled by batch_size todo # loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

                scaler.scale(loss * opt.beta).backward()

                if opt.use_hr and opt.use_lr:
                    lr_imgs = pad_lr(lr_imgs)
                    targets_lr = targets[:tn, :]
                    targets_lr[:, [2, 3]] = (targets_lr[:, [2, 3]] * imgsz // 4 + pl) / imgsz
                    targets_lr[:, [4, 5]] = (targets_lr[:, [4, 5]] * imgsz // 4) / imgsz

                    with amp.autocast():
                        pred = model(lr_imgs / 255.)
                    loss, loss_items = compute_loss(pred, targets_lr)
                    scaler.scale(loss * opt.beta).backward()

            #  Optimize
            scaler.step(optimizer)
            scaler.update()
            if opt.ema:
                modelema.update(model)

            #  logger loss && detail
            if epoch >= opt.epochs[0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # 当前 epoch 内已迭代 过的 batch 的平均损失

            if (i + 1) % opt.log_step == 0:
                logger.info('epoch {:>4}/{} | iter:{:>4}/{} | sr_loss {} | boxloss {} | objloss {} | clsloss {} | totalloss {} | labels {:<4} | time {}s | {}'
                            .format(epoch + 1, opt.epochs[1] + opt.epochs[0], i + 1, nb, '%.6f' % loss0, '%.6f' % mloss[0], '%.6f' % mloss[1], '%.6f' % mloss[2], '%.6f' % mloss[3],
                                    targets.shape[0], '%.4f' % (time.time() - t), 'only train D' if opt.only_det else 'Combine training' if epoch >= opt.epochs[0] else 'SR traing'))
                t = time.time()

            writer.add_scalars('det_loss', {'boxloss': mloss[0], 'objloss': mloss[1], 'clsloss': mloss[2], 'totalloss': mloss[3]}, epoch * nb + i + 1)
            writer.add_scalar('sr_loss', loss0, epoch * nb + i + 1)

        # logger & update lr
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch + 1)
        scheduler.step()
        if opt.ema:
            modelema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])

        # evaluate and save model
        is_final = (epoch + 1 == opt.epochs[1] + opt.epochs[0])
        avg_psnr = test_sr(model_sr, testloader, logger, cnt=640 if not is_final else 1e6)
        writer.add_scalar('psnr', avg_psnr, epoch + 1)

        if (is_final or epoch + 1 == opt.epochs[0]) and opt.epochs[1] != 0:
            logger.info('det test result in HR / LR(supered) images:')
            test(data_cfg, batch_size=opt.batch_size, imgsz=imgsz_test, model=modelema.ema if opt.ema else model,
                 model_sr=None, dataloader=testloader, save_dir=opt.save_dir, verbose=is_final,
                 plots=is_final, compute_loss=compute_loss, half=opt.test_half, logger=logger)
            results, maps, times = test(data_cfg, batch_size=opt.batch_size, imgsz=imgsz_test, model=modelema.ema if opt.ema else model,
                                        model_sr=model_sr, dataloader=testloader, save_dir=opt.save_dir, verbose=is_final,
                                        plots=is_final, compute_loss=compute_loss, half=opt.test_half, logger=logger)
            writer.add_scalars('det_eval', {'Precision': results[0], 'Recall': results[1], 'mAP@.5': results[2], 'mAP@.5:.95': results[3]}, epoch + 1)
        elif epoch >= opt.epochs[0] and (epoch - opt.epochs[0] + 1) % test_step == 0:
            results, maps, times = test(data_cfg, batch_size=opt.batch_size, imgsz=imgsz_test, model=modelema.ema if opt.ema else model,
                                        model_sr=model_sr, dataloader=testloader, save_dir=opt.save_dir, verbose=is_final,
                                        plots=is_final, compute_loss=compute_loss, half=opt.test_half, logger=logger)
            writer.add_scalars('det_eval', {'Precision': results[0], 'Recall': results[1], 'mAP@.5': results[2], 'mAP@.5:.95': results[3]}, epoch + 1)
        logger.info('\n')
        ckpt = {'epoch': epoch + 1,
                'model_sr': model_sr.state_dict(),
                'model': deepcopy(model).half(),
                'ema': deepcopy(modelema.ema).half(),
                'updates': modelema.updates,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
        if epoch + 1 == opt.epochs[0]:
            torch.save(ckpt, opt.save_dir / 'lastest_p1.pth')
        else:
            torch.save(ckpt, opt.save_dir / 'lastest_{}.pth'.format(epoch + 1))


def test_sr(model_sr, loader, logger, cnt=640, half=True):
    if half:
        model_sr.half()
    model_sr.eval()
    if half:
        model_sr.half()
    with torch.no_grad():
        # logger.info('Evaluating the PSNR-Y for {} images in VOC2007'.format(cnt))
        PSNR = 0
        cnt_ = 0
        for imgs, lr_imgs, _, _, _, in loader:
            imgs = imgs.cuda().half() if half else imgs.cuda().float()
            lr_imgs = lr_imgs.cuda().half() if half else imgs.cuda().float()
            sr_imgs = quantize(model_sr(lr_imgs))

            # import matplotlib.pyplot as plt
            # for idx in range(len(imgs)):
            #     sr = sr_imgs[idx].permute(1, 2, 0)
            #     hr = imgs[idx].permute(1, 2, 0)
            #     sr, hr = np.array(sr.cpu()).astype(np.uint8), np.array(hr.cpu()).astype(np.uint8)
            #     plt.imshow(sr)
            #     plt.show()
            #     plt.imshow(hr)
            #     plt.show()

            psnr_ = calc_psnr(imgs, sr_imgs, scale=4, benchmark=True)
            PSNR += psnr_.sum().item()
            cnt_ += imgs.shape[0]
            if cnt_ >= cnt:
                break
        avg_psnr = PSNR / cnt_
        logger.info('\033[1;31mThe avrage PSNR-Y for {} images in VOC2007 is {}\033[0m'.format(cnt_, '%.4f' % avg_psnr))
    model_sr.float()
    model_sr.train()
    return avg_psnr


def quantize(img, rgb_range=255):
    pixel_range = 255 / rgb_range
    img = img.mul_(pixel_range).clamp_(0, 255).round_().div_(pixel_range)
    return img


def calc_psnr(sr, hr, scale, rgb_range=255, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new_zeros(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow_(2).mean(-1).mean(-1).mean(-1)
    return -10 * torch.log10(mse)
