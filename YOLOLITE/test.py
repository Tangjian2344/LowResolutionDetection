from YOLOLITE.utils import non_max_suppression, scale_coords, xywh2xyxy, box_iou, ap_per_class, attempt_load, make_logger
import torch
from pathlib import Path
import numpy as np
from YOLOLITE.data import create_dataloader
import yaml
import cv2
import matplotlib.pyplot as plt


voc_cls = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus',
           6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse',
           13: 'motorbike', 14: 'person', 15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}


def test(data, batch_size=128, imgsz=416, model=None, model_sr=None, dataloader=None, save_dir='', verbose=False, plots=False, compute_loss=None,
         conf_thres=0.001, iou_thres=0.6, augment=False, save_txt=False, save_hybrid=False, half=True, logger=None, weights=None, test_lr=False, test_BISR=False):
    # Initialize/load model and set device
    training = model is not None
    test_sr = model_sr is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:
        device = torch.device('cuda')
        model = attempt_load(weights, device)  # load FP32 model
        with open('./data/voc.yaml') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)  # data dict todo 数据相关参数
        with open('./data/hyp.finetune.yaml') as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps  todo 训练的超参数
        dataloader = create_dataloader(data['val'], imgsz, batch_size, 32, hyp=hyp, rect=True, pad=0.5, prefix='val')[0]

    # Half
    # half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()
    if test_sr:
        model_sr.half()

    # Configure
    model.eval()
    if test_sr:
        model_sr.eval()
    # check_dataset(data)  # check
    nc = data['nc']  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    # confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # coco91class = coco80_to_coco91_class()
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    for batch_i, (img, lr_img, targets, paths, shapes) in enumerate(dataloader):
        lr_img = lr_img.to(device, non_blocking=True)
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        lr_img = lr_img.half() if half else lr_img.float()
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            # t = time_synchronized()

            if test_sr:
                img = model_sr(lr_img)
            elif test_lr:
                # lr_img_ = lr_img
                lr_img = torch.nn.ConstantPad2d(204, 114.)(lr_img)
                targets[:, [2, 3]] = (targets[:, [2, 3]] * 136 + 204) / 544
                targets[:, [4, 5]] = (targets[:, [4, 5]] * 136) / 544
                img = lr_img

                # for idx, sr in enumerate(img):
                #     sr = np.array(sr.cpu()).transpose(1, 2, 0).astype(np.uint8)
                #     sr = sr.copy()
                #     boxs = targets[targets[:, 0] == idx]
                #     H, W, _ = sr.shape
                #     for _, cls, x, y, w, h in boxs:
                #         x1, x2 = int(W * (x - w / 2)), int(W * (x + w / 2))
                #         y1, y2 = int(H * (y - h / 2)), int(H * (y + h / 2))
                #         cv2.rectangle(sr, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
                #     plt.imshow(sr)
                #     plt.show()
            elif test_BISR:
                img = []
                for path_sr in paths:
                    path_sr = path_sr.replace('test2007', 'test2007_BISR')
                    path_sr = path_sr.replace('pkl', 'jpg')
                    sr_img = cv2.imread(path_sr)
                    sr_img = sr_img[:, :, ::-1]
                    sr_img = sr_img.transpose(2, 0, 1)
                    sr_img = np.ascontiguousarray(sr_img)
                    sr_img = torch.from_numpy(sr_img).cuda().half()
                    img.append(sr_img)
                img = torch.stack(img, dim=0)
                img = torch.nn.ConstantPad2d(16, 144.)(img)

            out, train_out = model(img / 255.0, augment=augment)  # inference and training outputs
            # t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)

            # 可视化
            idxs = np.random.choice(len(img), len(img) // 16, replace=False)
            for idx in idxs:
                p_name = paths[idx].split('/')[-1].split('.')[0]
                p_target = np.array(targets[targets[:, 0] == idx].cpu())
                p_img = np.array(img[idx].cpu())
                p_img = np.transpose(p_img, (1, 2, 0)).astype(np.uint8).copy()
                p_out = out[idx]
                p_out = np.array(p_out[p_out[:, 4] > 0.2].cpu())

                # for _, cls, x, y, w, h in p_target:
                #     x1, y1, x2, y2 = [int(t) for t in [x - w / 2, y - h / 2, x + w / 2, y + h / 2]]
                #     cv2.rectangle(p_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                #     cv2.putText(p_img, voc_cls[int(cls)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                if test_lr:
                    p_img = cv2.resize(p_img, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
                    for x1, y1, x2, y2, conf, cls in p_out:
                        x1, y1, x2, y2 = [int(t) for t in [x1, y1, x2, y2]]
                        x1, y1, x2, y2 = x1 * 4, y1 * 4, x2 * 4, y2 * 4
                        cv2.rectangle(p_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(p_img, voc_cls[int(cls)] + ' ' + str(conf)[:4], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    p_img = p_img[816:1360, 816:1360, :]
                else:
                    for x1, y1, x2, y2, conf, cls in p_out:
                        x1, y1, x2, y2 = [int(t) for t in [x1, y1, x2, y2]]
                        cv2.rectangle(p_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(p_img, voc_cls[int(cls)] + ' ' + str(conf)[:4], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


                plt.imsave('./plots/NoiseSR/' + p_name + '.jpg', p_img)
        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    logger.info('\033[1;31mClass: ALL | Images: {} | Labels: {} | P: {} | R: {} | mAP@.5 {} | mAP@.5:.95 {}\033[0m'
                .format(seen, nt.sum(), '%.6f' % mp, '%.6f' % mr, '%.6f' % map50, '%.6f' % map))
    # pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format

    # Print results per class
    # if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
    #     logger.info('{:^12} {:^12} {:^12} {:^12} {:^12} {:^12}'.format('class', 'target', 'precision', 'recall', 'mAP@.5', 'mAP@.5:.95'))
    #     for i, c in enumerate(ap_class):
    #         logger.info('{:^12} {:^12} {:^12} {:^12} {:^12} {:^12}'.format(names[c], nt[c], '%.4f' % p[i], '%.4f' % r[i], '%.4f' % ap50[i], '%.4f' % ap[i]))
            # print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.6f/%.6f/%.6f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Return results
    model.float()  # for training
    model.train()
    if test_sr:
        model_sr.float()
        model_sr.train()
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


