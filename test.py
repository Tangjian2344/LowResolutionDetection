import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np

# path = '/home/tangjian/Codes/SRdataset/DIV2K/HR/'
# LR_path = '/home/tangjian/Codes/SRdataset/DIV2K/LR_BI/x4/'
#
#
#
# names = os.listdir(path)
# os.chdir(path)
#
# for name in names:
#     img = cv2.imread(name)
#     img = img.transpose(2, 0, 1)
#     img = torch.from_numpy(img).float().unsqueeze(0)
#     L_img = F.interpolate(img, size=(img.shape[2] // 4, img.shape[3]//4), mode='bicubic', align_corners=False)
#     a = np.array(L_img.squeeze(0)).transpose(1, 2, 0)
#     L_img = np.array(L_img.squeeze(0)).transpose(1, 2, 0).round().clip(0, 255).astype(np.uint8)
#     cv2.imwrite(LR_path + name.split('.')[0] + 'x4.png', L_img)
#     # pass


# B = ['B100', 'Manga109', 'Set14', 'Set5', 'Urban100']
# path_ = '/home/tangjian/Codes/SRdataset/benchmark/HR/'
# path_l = '/home/tangjian/Codes/SRdataset/benchmark/LR_BI/'
# for b in B:
#     path = path_ + b + '/X4'
#     LR_path = path_l + b + '/x4/'
#     names = os.listdir(path)
#     os.chdir(path)
#
#     for name in names:
#         img = cv2.imread(name)
#         img = img.transpose(2, 0, 1)
#         img = torch.from_numpy(img).float().unsqueeze(0)
#         L_img = F.interpolate(img, size=(img.shape[2] // 4, img.shape[3]//4), mode='bicubic', align_corners=False)
#         a = np.array(L_img.squeeze(0)).transpose(1, 2, 0)
#         L_img = np.array(L_img.squeeze(0)).transpose(1, 2, 0).round().clip(0, 255).astype(np.uint8)
#         name_l = name.replace('HR', 'LRBI')
#         cv2.imwrite(LR_path + name_l, L_img)
#         pass

# for idx, sr in enumerate(lr_imgs):
#     sr = np.array(sr.cpu()).transpose(1, 2, 0).astype(np.uint8)
#     sr = sr.copy()
#     boxs = targets_lr[targets_lr[:, 0] == idx]
#     H, W, _ = sr.shape
#     for _, cls, x, y, w, h in boxs:
#         x1, x2 = int(W * (x - w / 2)), int(W * (x + w / 2))
#         y1, y2 = int(H * (y - h / 2)), int(H * (y + h / 2))
#         cv2.rectangle(sr, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
#     plt.imshow(sr)
#     plt.show()


#  计算BICUBIC 的SR

def calc_psnr(sr, hr, scale=4, rgb_range=255, benchmark=True):
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

# import glob
# import torch
# import matplotlib.pyplot as plt
#
# sr_path = '/home/tangjian/Codes/VOC/images512/test2007_BISR/'
# hr_path = '/home/tangjian/Codes/VOC/images512/test2007/'
#
# sr_imgs = glob.glob(sr_path + '*.jpg')
# sr_imgs.sort()
# hr_imgs = glob.glob(hr_path + '*.jpg')
# hr_imgs.sort()
# PSNR = []
# for srf, hrf in zip(sr_imgs, hr_imgs):
#     sr = cv2.imread(srf).transpose(2, 0, 1)[::-1, :, :]
#     sr = torch.from_numpy(np.ascontiguousarray(sr)).unsqueeze(0).float()
#     hr = cv2.imread(hrf).transpose(2, 0, 1)[::-1, :, :]
#     hr = torch.from_numpy(np.ascontiguousarray(hr)).unsqueeze(0).float()
#     psnr = calc_psnr(sr, hr)
#     PSNR.append(psnr.item())
# print(sum(PSNR) / len(PSNR))





os.chdir('/home/tangjian/Codes/')
train_idx = np.load('trainidxs1920.npy')
test_idx = np.load('validxs1920.npy')
print(test_idx)