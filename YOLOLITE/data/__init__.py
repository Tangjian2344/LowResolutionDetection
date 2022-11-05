import torch
from YOLOLITE.data.oddata import LoadImagesAndLabels
from torch.utils.data import DataLoader


def create_dataloader(path, imgsz, batch_size, stride, hyp=None, augment=False, pad=0.0, rect=False, workers=8, image_weights=False, quad=False, prefix='', combine=False, idxs1920=None):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache

    dataset = LoadImagesAndLabels(path, imgsz, batch_size, augment=augment, hyp=hyp, rect=rect,
                                  stride=stride, pad=pad, image_weights=image_weights, prefix=prefix, combine=combine, idxs1920=idxs1920)

    batch_size = min(batch_size, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, shuffle=True if prefix=='train' else False,
                            collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn if combine else LoadImagesAndLabels.collate_fn1)
    return dataloader, dataset


if __name__ == '__main__':
    import yaml
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    voc_cls = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
               'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 29}

    with open('hyp.finetune.yaml') as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    opt = {'single_cls': False}
    path = 'F:/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    img_size = 512
    batch_size = 4
    stride = 32
    augment = True
    rect = False
    image_weights = False
    cache_images = False
    single_cls = False
    pad = 0.0
    prefix = ''
    dataloader, dataset = create_dataloader(path, img_size, batch_size, stride, opt, hyp)

    for d in dataloader:
        break
