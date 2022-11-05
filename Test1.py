import argparse
import time
from YOLOLITE.loss import ComputeLoss
import yaml
from YOLOLITE.utils import make_logger
import torch
from YOLOLITE.data import create_dataloader
from YOLOLITE.model.yololite import Model
from LROD.model import make_model
from YOLOLITE.test import test
from YOLOLITE.model.common import ModelEMA
import os
from Bicubic import BICUBIC
import random
import numpy as np
import torch.backends.cudnn as cudnn
from Trainer1 import test_sr


parser = argparse.ArgumentParser(description='Train the LROD model')
parser.add_argument('--config_sr', type=str, default='./LROD/config/config_sr.yaml', help='configuration file for the SR subnetwork')
parser.add_argument('--test_only', action='store_true', help='Is it just testing?')
parser.add_argument('--test_model', type=str, default='', help='model to test')
parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale')
# parser.add_argument('--workers', type=int, default=4, help='Loading data thread')
# parser.add_argument('--out_path', type=str, default='./exp/output', help='dir to save output during training phase')
parser.add_argument('--log_step', type=int, default=10, help='frequency of save log during an epoch')
parser.add_argument('--save_freq', type=int, default=10, help='frequency of save checkpoint during whole training phase')
parser.add_argument('--pass1', action='store_true', help='Is it step the phase of traing sr along')
parser.add_argument('--weight_sr_fined', type=str, default="./exped/output_2022_04_05_16_30_26_1.0_1.0/lastest_p1.pth", help='f weights path')

parser.add_argument('--resume', action='store_true', help='Is it continue train from last time')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file need to be loaded')

parser.add_argument('--pretrained', action='store_false', help='Whether to load the pretrained model')
parser.add_argument('--weight_od', type=str, default='./YOLOLITE/save_nowarm/lastest.pth', help='initial weights path')
parser.add_argument('--weight_sr', type=str, default='./LROD/exp/output_baseline/model_latest.pth', help='initial weights path')
parser.add_argument('--cfg', type=str, default='./YOLOLITE/model/v5Lite-s.yaml', help='model.yaml path')
parser.add_argument('--data', type=str, default='./YOLOLITE/data/voc.yaml', help='data.yaml path')
parser.add_argument('--hyp', type=str, default='./YOLOLITE/data/hyp.finetune.yaml', help='hyperparameters path')
parser.add_argument('--epochs', type=int, default=[25, 25])
parser.add_argument('--test_freq', type=int, default=5, help='the frequence for evaluate model in test dataset during training phase')
parser.add_argument('--ema', action='store_false', help='Model Exponential Moving Average')
parser.add_argument('--test_half', action='store_false', help='Test with half precision')
parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all GPUs')
parser.add_argument('--img_size', nargs='+', type=int, default=[512, 512], help='[train, test] image sizes')
parser.add_argument('--rect', action='store_true', help='rectangular training')
# parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
parser.add_argument('--image_weights', action='store_true', help='use weighted image selection for training')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
parser.add_argument('--linear-lr', action='store_true', help='linear LR')
parser.add_argument('--quad', action='store_true', help='quad dataloader')
parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
parser.add_argument('--save_dir', type=str, default="./test/", help='the dir to save output during traing/testing phase')
parser.add_argument('--combine', action='store_false', help='is trained combine sr and od')
parser.add_argument('--number', type=str, default='_', help='choose a modelsr to run')
parser.add_argument('--degra', type=str, default='')
parser.add_argument('--test_n', type=int, default=0)

opt = parser.parse_args()
with open(opt.hyp) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
with open(opt.data) as f:
    data_cfg = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
with open(opt.cfg) as f:
    model_cfg = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
with open(opt.config_sr) as f:
    config_sr = yaml.load(f, Loader=yaml.SafeLoader)

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = True, False

init_seeds()


imgsz, imgsz_test = opt.img_size
is_final = True
gs = 32

bicubic = BICUBIC()


# path = paths[opt.test_n]

# print('*************eval result in {}  ******************'.format(path))
opt.degra = 'noise'
hyp['degra'] = opt.degra

# ckptpath = "/home/tangjian/Codes/expnew/output_2022_04_26_04_48_36_0.0_1.0/lastest_50.pth"  # baseline D
# ckptpath = "/home/tangjian/Codes/expnew/output_2022_04_26_06_21_07_1.0_0.1/lastest_50.pth"  # ours
# ckptpath = "/home/tangjian/Codes/expnew/output_2022_04_28_00_20_25_1.0_0.1/lastest_50.pth"  # blur
""""""
ckptpath = "/home/tangjian/Codes/expnew/output_2022_04_28_06_07_21_1.0_0.1/lastest_50.pth"  # noise
ckpt = torch.load(ckptpath)
opt.save_dir = os.path.dirname(ckptpath)
logger = make_logger(log_file=opt.save_dir, test_only=True)

model = Model(model_cfg, ch=3, nc=20).cuda()
modelema = ModelEMA(model)
model.load_state_dict(ckpt['model'].float().state_dict())
if opt.ema:
    modelema.ema.load_state_dict(ckpt['ema'].float().state_dict())
    modelema.updates = ckpt['updates']

ckptpath = "/home/tangjian/Codes/exp/output_2022_04_09_21_12_00_1.0_0/lastest.pth"
# ckpt = torch.load(ckptpath)

model_sr = None
if ckpt.get('model_sr'):
    model_sr = make_model(config_sr['Model'], opt).cuda()
    model_sr.load_state_dict(ckpt['model_sr'])

model.gr = 1.0
model.hyp = hyp
compute_loss = ComputeLoss(model)
# test_idx = np.load('validxs1920.npy')
test_idx = None

# hyp['eval_params'] = True
testloader = create_dataloader(data_cfg['val'], imgsz_test, opt.batch_size, gs, hyp=hyp, rect=True, pad=0.5, prefix='val', combine=opt.combine, idxs1920=test_idx) [0]


# logger.info('test result in LR images')
# results, maps, times = test(data_cfg, batch_size=opt.batch_size, imgsz=imgsz_test, model=modelema.ema if opt.ema else model,
#                             model_sr=None, dataloader=testloader, save_dir=opt.save_dir, verbose=is_final,
#                             plots=is_final, compute_loss=compute_loss, half=opt.test_half, logger=logger, test_lr=True)


if model_sr:
    logger.info('test result in LR images after super-resolution:')
    results, maps, times = test(data_cfg, batch_size=opt.batch_size, imgsz=imgsz_test, model=modelema.ema if opt.ema else model,
                                model_sr=model_sr, dataloader=testloader, save_dir=opt.save_dir, verbose=is_final,
                                plots=is_final, compute_loss=compute_loss, half=opt.test_half, logger=logger)


# logger.info('test result in HR images')
# results, maps, times = test(data_cfg, batch_size=opt.batch_size, imgsz=imgsz_test, model=modelema.ema if opt.ema else model,
#                             model_sr=None, dataloader=testloader, save_dir=opt.save_dir, verbose=is_final,
#                             plots=is_final, compute_loss=compute_loss, half=opt.test_half, logger=logger)
#
if model_sr:
    logger.info('test PSNR ...')
    test_sr(model_sr, testloader, logger, 1e6)

