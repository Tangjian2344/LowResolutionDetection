from LROD.model import make_model
from LROD.data import make_dataloader
from LROD.trainer import Trainer
from LROD.utility import make_logger, record_param
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
import time

parser = argparse.ArgumentParser(description='Train the LROD model')
parser.add_argument('--config_sr', type=str, default='./LROD/config/config_sr.yaml', help='configuration file for the SR subnetwork')
parser.add_argument('--test_only', action='store_true', help='Is it just testing?')
parser.add_argument('--test_model', type=str, default='', help='model to test')
parser.add_argument('--scale', type=int, default=3, help='Super-resolution scale')
parser.add_argument('--workers', type=int, default=4, help='Loading data thread')
parser.add_argument('--GPU', action='store_false', help='Whether to use GPU')
parser.add_argument('--batch', type=int, default=16, help='Training batchSize')
parser.add_argument('--epoch', type=int, default=150, help='Training epochs')
parser.add_argument('--out_path', type=str, default='./LROD/expnew/output', help='dir to save output during training phase')
parser.add_argument('--log_freq', type=int, default=20, help='frequency of save log during an epoch')
parser.add_argument('--save_freq', type=int, default=20, help='frequency of save checkpoint during whole training phase')
parser.add_argument('--resume', action='store_true', help='Is it continue train from last time')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file need to be loaded')
parser.add_argument('--number', type=str, default='_', help='choose a modelsr to run')


def run():
    args = parser.parse_args()
    with open(args.config_sr, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['Dataset']['train']['patch_size'] = 48 * args.scale

    if args.test_only:  # 仅测试
        args.out_path = os.path.dirname(args.test_model)
    elif args.resume:  # 断点训练
        args.out_path = os.path.dirname(args.checkpoint)
    else:
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        args.out_path = args.out_path + '_' + cur_time
        os.mkdir(args.out_path)

    logger = make_logger(mode='a' if args.resume else 'w', log_file=args.out_path, test_only=args.test_only)

    if args.test_only:  # 仅测试
        logger.info('it will test the model in {}'.format(args.test_model))
    elif args.resume:
        logger.info('it will load checkpoint file from {}'.format(args.checkpoint))
    else:
        logger.info(record_param({'The training configuration is as follows': {'Configs': config, 'Args': vars(args)}}))

    model = make_model(config['Model'], args).cuda()

    if not args.test_only:
        writer = SummaryWriter(log_dir=args.out_path)
        loader_dic = make_dataloader(config['Dataset']['train'], args)
        trainer = Trainer(args, loader_dic, model, logger, config['Train'], writer)
        trainer.train()
    else:
        loader_dic = make_dataloader(config['Dataset']['test'], args)
        trainer = Trainer(args, loader_dic, model, logger)
        trainer.test()
