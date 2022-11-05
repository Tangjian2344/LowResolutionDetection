import argparse
import os
import yaml
import time
from LROD.utility import make_logger, record_param
from Trainer1 import train
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

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
parser.add_argument('--weight_sr_fined', type=str, default='', help='f weights path')
parser.add_argument('--number', type=str, default='_', help='choose a modelsr to run')
parser.add_argument('--resume', action='store_true', help='Is it continue train from last time')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file need to be loaded')

parser.add_argument('--pretrained', action='store_false', help='Whether to load the pretrained model')
parser.add_argument('--weight_od', type=str, default='./YOLOLITE/save_nowarm/lastest.pth', help='initial weights path')
parser.add_argument('--weight_sr', type=str, default='./LROD/expnew/outputX4/model_latest.pth', help='initial weights path')
parser.add_argument('--cfg', type=str, default='./YOLOLITE/model/v5Lite-s.yaml', help='model.yaml path')
parser.add_argument('--data', type=str, default='./YOLOLITE/data/voc.yaml', help='data.yaml path')
parser.add_argument('--hyp', type=str, default='./YOLOLITE/data/hyp.finetune.yaml', help='hyperparameters path')
parser.add_argument('--epochs',type=int, nargs='+', default=[25, 25])
parser.add_argument('--test_freq', type=int, default=10, help='the frequence for evaluate model in test dataset during training phase')
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
parser.add_argument('--save_dir', type=str, default="./exp/debug", help='the dir to save output during traing/testing phase')
parser.add_argument('--combine', action='store_false', help='is trained combine sr and od')
parser.add_argument('--alpha', type=float, default=0.)
parser.add_argument('--beta', type=float, default=0.)
parser.add_argument('--use_hr', action='store_true', help='is used hr img to traing det net dur conbine traing phase')
parser.add_argument('--use_lr', action='store_true', help='is used lr img to traing det net dur conbine traing phase')
# parser.add_argument('--lr_scale', type=float, default=1.)
parser.add_argument('--degra', type=str, default='')
parser.add_argument('--only_det', action='store_true')

opt = parser.parse_args()
#
with open(opt.hyp) as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
with open(opt.data) as f:
    data_cfg = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
with open(opt.cfg) as f:
    model_cfg = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
# sr 参数
with open(opt.config_sr) as f:
    config_sr = yaml.load(f, Loader=yaml.SafeLoader)

hyp['degra'] = opt.degra


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = True, False

init_seeds()


if opt.test_only:  # 仅测试
    opt.save_dir = os.path.dirname(opt.test_model)
elif opt.resume:  # 断点训练
    opt.save_dir = os.path.dirname(opt.checkpoint)
else:
    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    opt.save_dir = opt.save_dir + '_' + cur_time + '_' +  str(opt.alpha) + '_' + str(opt.beta)
    # if opt.lr_scale:
    #     opt.save_dir = opt.save_dir + '_' + str(opt.lr_scale)
    os.mkdir(opt.save_dir)

logger = make_logger(mode='a' if opt.resume else 'w', log_file=opt.save_dir, test_only=opt.test_only)
writer = SummaryWriter(log_dir=opt.save_dir)
train(hyp, data_cfg, model_cfg, opt, config_sr, logger, writer)