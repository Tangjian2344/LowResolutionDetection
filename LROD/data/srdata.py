import os
from LROD.data import common
import pickle
import torch.utils.data as data
from PIL import Image
import numpy as np


class SRData(data.Dataset):
    def __init__(self, config, args, test_only=False):
        self.config = config
        self.name = config['name']
        self.test_only = test_only
        self.scale = args.scale

        self._set_filesystem(config['dir'])
        self.list_hr, self.list_lr, self.img_nums = self._scan()
        self.repeat = config['test_every'] // (self.img_nums // args.batch) if not test_only else 1

    def _scan(self):
        names_hr, names_lr = sorted(os.listdir(self.dir_hr)), sorted(os.listdir(self.dir_lr))
        dir_hr_bin, dir_lr_bin = self.dir_hr.replace(self.name, self.name + '_bin'), self.dir_lr.replace(self.name, self.name + '_bin')
        os.makedirs(dir_hr_bin, exist_ok=True)
        os.makedirs(dir_lr_bin, exist_ok=True)
        for i, (name_hr, name_lr) in enumerate(zip(names_hr, names_lr)):
            bin_hr_file = os.path.join(dir_hr_bin, name_hr.split('.')[0] + '.pkl')
            bin_lr_file = os.path.join(dir_lr_bin, name_lr.split('.')[0] + '.pkl')
            common.save_bin_file(bin_hr_file, os.path.join(self.dir_hr, name_hr))
            common.save_bin_file(bin_lr_file, os.path.join(self.dir_lr, name_lr))
            names_hr[i], names_lr[i] = bin_hr_file, bin_lr_file
        return names_hr, names_lr, len(names_hr)

    def _set_filesystem(self, dir_data):
        apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(apath, 'HR')
        self.dir_lr = os.path.join(apath, 'LR_BI', 'X{}'.format(self.scale))

    def __getitem__(self, idx):
        idx = idx % self.img_nums
        hr_file, lr_file = self.list_hr[idx], self.list_lr[idx]
        if not self.test_only:
            with open(hr_file, 'rb') as f:
                hr = pickle.load(f)
            with open(lr_file, 'rb') as f:
                lr = pickle.load(f)
            lr, hr = self.get_patch(lr, hr)
        else:
            hr = np.array(Image.open(hr_file))
            lr = np.array(Image.open(lr_file))
        lr, hr = common.set_channel(lr, hr)
        lr, hr = common.np2Tensor(lr, hr)

        return lr, hr, idx, hr_file

    def __len__(self):
        return self.img_nums * self.repeat

    def get_patch(self, lr, hr):
        if not self.test_only:
            lr, hr = common.get_patch(lr, hr, patch_size=self.config['patch_size'],  scale=self.scale)
            if self.config['augment']:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * self.scale, 0:iw * self.scale]
        return lr, hr

