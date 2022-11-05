import numpy as np
import cv2
import os

name_num = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
            'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
num_name = {v: k for k, v in name_num.items()}


def analyze_data(dataloader, flag='train', dir="/home/tangjian/Codes/analyze_data/", add_box=True):
    cls_nums = [0] * 20
    for i, (imgs, lr_imgs, targets, paths, _) in enumerate(dataloader):

        for i in range(20):
            cls_nums[i] += (targets[:, 1] == i).sum().item()
        n = len(imgs)
        idxs = np.random.choice(n, n//10, replace=False)
        for idx in idxs:
            lr = np.array(lr_imgs[idx]).transpose(1, 2, 0).astype(np.uint8)
            hr = np.array(imgs[idx]).transpose(1, 2, 0).astype(np.uint8)
            hr = hr.copy()
            H, W, _ = hr.shape
            if add_box:
                boxs = targets[targets[:, 0] == idx]
                for _, cls, x, y, w, h in boxs:
                    x1, x2 = int(W * (x - w / 2)), int(W * (x + w / 2))
                    y1, y2 = int(H * (y - h / 2)), int(H * (y + h / 2))
                    cv2.rectangle(hr, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
                    cv2.putText(hr, num_name[int(cls)], (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            H_, W_, _ = lr.shape
            bt, bd = (H - H_) // 2, (H - H_) // 2
            bl, br = (W - W_) // 2, (W - W_) // 2
            lr = cv2.copyMakeBorder(lr, bt, bd, bl, br,  cv2.BORDER_CONSTANT, value=(114, 114, 114))
            res = np.hstack((lr, hr))
            src_path = os.path.join(dir, flag) + '/'
            res = res[:, :, ::-1]

            cv2.imwrite(src_path + os.path.basename(paths[idx]).replace('pkl', 'jpg'), res)
    print('the box num of  each class in selected {} data {}'.format(flag, cls_nums))


