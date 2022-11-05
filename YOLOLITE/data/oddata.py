import pathlib as Path
import tqdm
import cv2
import torch
from pathlib import Path
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
import xml.etree.ElementTree as ET
from YOLOLITE.data.common import xywhn2xyxy, xyxy2xywh
import math
import pickle
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import util


img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes


voc_cls = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
           'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}


for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def get_xml(file, shape, imgfile=None):
    # shape: h, w
    tree = ET.parse(file)
    root = tree.getroot()

    objs = root.findall('./object')
    ans = []
    for obj in objs:
        cls = obj.find('./name').text
        box = obj.find('./bndbox')
        xmin, ymin, xmax, ymax = map(int,[box.find(c).text for c in ['xmin', 'ymin', 'xmax', 'ymax']])
        assert xmax > xmin and ymax > ymin, 'voc 坐标格式有误'
        xmin, xmax, ymin, ymax = xmin/shape[1], xmax/shape[1], ymin/shape[0], ymax/shape[0]
        ans.append([voc_cls[cls], xmin, ymin, xmax, ymax])
    assert ans, f'there is no box in {file}'

    if imgfile and random.random() > 0.995:
        with open(imgfile, 'rb') as f:
            img = pickle.load(f)
        for cls, x1, y1, x2, y2 in ans:
            cv2.rectangle(img, (int(x1 * shape[1]), int(y1 * shape[0])), (int(x2 * shape[1]), int(y2 * shape[0])), (255, 0, 0), 2)
            # cv2.putText(hr_img, )
        plt.imshow(img)
        plt.show()

    ans = np.array(ans, dtype=np.float32)
    ans[:, 1:] = xyxy2xywh(ans[:, 1:])
    return ans


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False,
                 image_weights=False, stride=32, pad=0.0, prefix='', combine=True, idxs1920=None):
        self.combine = combine
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights  # todo False
        self.rect = False if image_weights else rect  # todo False
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.prefix = prefix

        self.label_files = []  # 保持 每张图片 的 标签 xml  文件路径
        path = [path] if not isinstance(path, list) else path
        for file in path:
            file = Path(file)
            label_txt = (file.parent / 'labels' / file.name).with_suffix('.txt')
            with label_txt.open('r') as f:
                names = f.readlines()
            self.label_files.extend([(file.parent / 'labels' / file.name) / (base_name.rstrip() + '.xml') for base_name in names])
        del file, f, names, label_txt
        self.img_files = ['images512'.join(str(xml.with_suffix('.pkl')).split('labels')) for xml in self.label_files]
        self.label_files = [str(x) for x in self.label_files]
        if self.combine:
            def rep(x):
                if 'train2012' in x:
                    return x.replace('train2012', 'train2012_lr')
                elif 'train2007' in x:
                    return x.replace('train2007', 'train2007_lr')
                elif 'test2007' in x:
                    return x.replace('test2007', 'test2007_lr')
                else:
                    raise('{}不包含 train2012/train2007/test2007'.format(x))
            self.lr_img_files = [rep(x) for x in self.img_files]

        cache_path = Path(self.label_files[0]).parents[1] / (prefix + '.cache')  # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
                cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            print("Scanning '{}' images and labels {} found, {} missing, {} empty, {} corrupted".format(cache_path, nf, nm, ne, nc))
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See help_url'

        # Read cache  todo cache
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())  # box cords, shape,
        self.labels = list(labels)  # 数据集中每一张图片的标签
        self.shapes = np.array(shapes, dtype=np.float64)  # 数据集中每一张图片的大小
        self.img_files = list(cache.keys())  # 利用 cache 中数据集图片名更新 之前的数据集图片名
        # self.label_files = img2label_paths(cache.keys())  # update todo 注释
        self.label_files = [('labels'.join(x.split('images'))).split('.')[0] + '.xml' for x in self.img_files]
        if hyp['eval_params']:
            idxs = np.random.choice(n, hyp['select_num'], replace=False) if idxs1920 is None else idxs1920
            # print(idxs)
            # np.save(prefix + 'idxs1920.npy', idxs)
            self.img_files = [self.img_files[i] for i in idxs]
            self.lr_img_files = [self.lr_img_files[i] for i in idxs]
            self.label_files = [self.label_files[i] for i in idxs]
            self.labels = [self.labels[i] for i in idxs]
            self.shapes = self.shapes[idxs]
            shapes = [shapes[i] for i in idxs]
            n = len(idxs)

        # import matplotlib.pyplot as plt
        # for i, path in enumerate(self.img_files):
        #     with open(path, 'rb') as f:
        #         hr_img = pickle.load(f)
        #     boxes = self.labels[i]
        #     H, W, _ = hr_img.shape
        #     for cls, x, y, w, h in boxes:
        #         x1, x2 = int((x - w / 2) * W), int((x + w / 2) * W)
        #         y1, y2 = int((y - h / 2) * H), int((y + h / 2) * H)
        #         cv2.rectangle(hr_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #         # cv2.putText(hr_img, )
        #     plt.imshow(hr_img)
        #     plt.show()
        #     continue

        self.n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.indices = range(n)
        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio  高宽比
            if (ar != 1.).sum() > 0:
                irect = ar.argsort()  # 高宽比排序后的索引
                self.img_files = [self.img_files[i] for i in irect]
                self.label_files = [self.label_files[i] for i in irect]
                self.labels = [self.labels[i] for i in irect]
                self.shapes = s[irect]  # wh
                ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride
        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im_file0 = im_file.split('.')[0] + '.jpg'
                im = Image.open(im_file0)
                im.verify()  # PIL verify  todo 检查文件完整性
                shape = exif_size(im)  # image size
                segments = []  # instance segments
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    l = get_xml(lb_file, shape, im_file)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'  # todo 负样本
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'  # todo box 坐标没用归一化
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'  # todo 一张图上有相同的标签
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]  # todo key:value  图片名：[图片标签array, 形状, ]
            except Exception as e:
                nc += 1
                print(f'{prefix} WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f"{prefix} Scanning '{path.parent / path.stem}' images and labels " f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()

        if nf == 0:
            print(f'{prefix} WARNING: No labels found in {path}. See help_url')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        # logging.info(f'{prefix}New cache created: {path}')
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        # 读取低分辨率图片
        if self.combine:
            with open(self.lr_img_files[index], 'rb') as f:
                lr_img = pickle.load(f)

            if self.hyp['degra'] == 'blur':
                lr_img = cv2.GaussianBlur(lr_img, ksize=(7, 7), sigmaX=0)
            elif self.hyp['degra'] == 'noise':
                lr_img = util.random_noise(lr_img, mode='gaussian', var=0.01)*255

            assert lr_img is not None, 'Image Not Found ' + self.lr_img_files[index]
            if self.rect:
                lr_img = cv2.copyMakeBorder(lr_img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            lr_img = lr_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            lr_img = np.ascontiguousarray(lr_img)

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic and not self.combine:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, self.n - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()

            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])  # pad 后图像上的绝对坐标 x1 y1 x2 y2

        if self.augment and not self.combine:
            # Augment imagespace
            if not mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment and not self.combine:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        # 可视化一下
        # H, W, _ = img.shape
        # for cls, x, y, w_, h_ in labels:
        #     x1, x2 = x - w_ / 2, x + w_ / 2
        #     y1, y2 = y - h_ / 2, y + h_ / 2
        #     cv2.rectangle(img, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (255, 0, 0), 2)
        # plt.imshow(img)
        # plt.show()

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)  # 标签 重 n x 5 ---> n x 6  多出来的一个元素用来记录 所在批次的索引

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        # mylabels = labels_out[:, 2:] * 416
        # mylabels[:, :2] = mylabels[:, :2] - mylabels[:, 2:] / 2
        # plt.figure()
        # plt.imshow(img.transpose(1, 2, 0))
        # currentAxis = plt.gca()
        # for x, y, w, h in mylabels:
        #     rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        #     currentAxis.add_patch(rect)
        # plt.show()
        if self.combine:
            return torch.from_numpy(img), torch.from_numpy(lr_img), labels_out, self.img_files[index], shapes
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, lr_img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.stack(lr_img, 0), torch.cat(label, 0), path, shapes
        # img, label, path, shapes = zip(*batch)  # transposed
        # for i, l in enumerate(label):
        #     l[:, 0] = i  # add target image index for build_targets()
        # return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn1(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    img = self.imgs[index]

    # todo 用于联合训练的数据加载，不做缩放
    if self.combine:
        path = self.img_files[index]
        # img = cv2.imread(path)  # BGR
        # path = path.replace('test2007', 'test2007_BISR')
        with open(path, 'rb') as f:
            img = pickle.load(f)

        # if self.prefix == 'val':
        #     if self.hyp['degra'] == 'blur':
        #         img = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=0)
        #     elif self.hyp['degra'] == 'noise':
        #         img = util.random_noise(img, mode='gaussian', var=0.01) * 255

        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)
        if r == 1:
            return img, (h0, w0), img.shape[:2]
        else:
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)  # 图片的长边缩放到 给定大小 如416 纵横比保持不变
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    # loads 1 image from dataset, returns img, original hw, resized hw
    if img is None:  # not cached
        path = self.img_files[index]
        # img = cv2.imread(path)  # BGR
        with open(path, 'rb') as f:
            img = pickle.load(f)
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)  # 图片的长边缩放到 给定大小 如416 纵横比保持不变
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def hist_equalize(img, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def load_mosaic(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size  # 超参数 train /test 图像大小
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y   [img_size/2 --> img_size * 3 / 2]
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):  # 遍历选中的四张图片
        # Load image
        img, _, (h, w) = load_image(self, index)  # 缩放后的图像，原图像大小，缩放后的大小

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles  (416*2, 416*2, 3)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates






