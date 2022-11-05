import torch
import logging
import os


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
    mse = valid.pow_(2).mean()
    return -10 * torch.log10(mse)


def make_logger(mode='w', log_file='', test_only=False):
    log_file = os.path.join(log_file, 'train.log' if not test_only else 'test.log')

    logger = logging.getLogger('RLOD')
    logger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename=log_file, mode=mode)

    formatter_c = logging.Formatter(fmt='%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # formatter_f = logging.Formatter(fmt='%(asctime)s | %(levelname)s | %(filename)s | %(lineno)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    consoleHandler.setFormatter(formatter_c)
    fileHandler.setFormatter(formatter_c)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger


def record_param(dic_p, t=0, width=32):
    param = ''
    for k, v in dic_p.items():
        if isinstance(v, dict):
            param += '{}{}:\n{}'.format(t * '\t', k, record_param(v, t + 1, width - 4))

        else:
            black = (width - len(k))
            param += '{}{}{}\n'.format(t*'\t', k + ':' + ' ' * black, str(v))
    return param



