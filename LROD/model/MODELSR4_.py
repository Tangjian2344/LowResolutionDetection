import torch
import torch.nn as nn

#1
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ARB(nn.Module):
    def __init__(self, n_feats, kernel_size, block_feats, wn, act=nn.ReLU(True)):
        super(ARB, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        # body = []
        # body.append(wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size // 2)))
        # body.append(act)
        # body.append(wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size // 2)))

        self.body = nn.Sequential(wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size // 2)),
                                  act,
                                  wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size // 2)))

    def forward(self, x):
        res = self.res_scale(self.body(x)) + self.x_scale(x)
        return res


class MUSM(nn.Module):
    def __init__(self, n_colors, scale, n_feats, wn):
        super(MUSM, self).__init__()
        out_feats = scale * scale * n_colors
        self.tail_k3 = wn(nn.Conv2d(n_feats, out_feats, kernel_size=3, padding=3 // 2, dilation=1))
        # self.tail_k5 = wn(nn.Conv2d(n_feats, out_feats, kernel_size=3, padding=5 // 2, dilation=2))
        # self.tail_k7 = wn(nn.Conv2d(n_feats, out_feats, kernel_size=3, padding=7 // 2, dilation=3))
        # self.tail_k9 = wn(nn.Conv2d(n_feats, out_feats, kernel_size=3, padding=9 // 2, dilation=4))
        # self.tail_k5 = wn(nn.Conv2d(n_feats, out_feats, 5, padding=5 // 2, dilation=1))
        # self.tail_k7 = wn(nn.Conv2d(n_feats, out_feats, 7, padding=7 // 2, dilation=1))
        # self.tail_k9 = wn(nn.Conv2d(n_feats, out_feats, 9, padding=9 // 2, dilation=1))
        self.pixelshuffle = nn.PixelShuffle(scale)
        # self.scale_k3 = Scale(0.25)
        # self.scale_k5 = Scale(0.25)
        # self.scale_k7 = Scale(0.25)
        # self.scale_k9 = Scale(0.25)

    def forward(self, x):
        # x0 = self.pixelshuffle(self.scale_k3(self.tail_k3(x)))
        # x1 = self.pixelshuffle(self.scale_k5(self.tail_k5(x)))
        # x2 = self.pixelshuffle(self.scale_k7(self.tail_k7(x)))
        # x3 = self.pixelshuffle(self.scale_k9(self.tail_k9(x)))
        #
        # return x0 + x1 + x2 + x3
        return self.pixelshuffle(self.tail_k3(x))

class IARB(nn.Module):
    def __init__(self, n_feats, kernel_size, block_feats, wn, act=nn.ReLU(True)):
        super(IARB, self).__init__()
        self.b0 = ARB(n_feats, kernel_size, block_feats, wn=wn, act=act)
        self.b1 = ARB(n_feats, kernel_size, block_feats, wn=wn, act=act)
        self.b2 = ARB(n_feats, kernel_size, block_feats, wn=wn, act=act)
        self.b3 = ARB(n_feats, kernel_size, block_feats, wn=wn, act=act)
        self.reduction = wn(nn.Conv2d(n_feats * 4, n_feats, 1, padding=0))
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        self.scale1 = Scale(1)
        self.scale2 = Scale(1)
        self.scale3 = Scale(1)
        self.scale4 = Scale(1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        res = self.reduction(torch.cat([self.scale1(x0), self.scale1(x1), self.scale1(x2), self.scale1(x3)], dim=1))

        return self.res_scale(res) + self.x_scale(x)


class MODEL(nn.Module):
    def __init__(self, config, args):
        super(MODEL, self).__init__()
        # hyper-params
        # self.args = args
        scale = args.scale
        n_resblocks = config['n_resblocks']
        n_feats = config['n_feats']
        kernel_size = 3
        act = nn.ReLU(True)

        # wn = lambda x: x
        def wn(x): return torch.nn.utils.weight_norm(x)

        # self.rgb_mean = torch.autograd.Variable(torch.FloatTensor([0.4488, 0.4371, 0.4040])).view([1, 3, 1, 1])
        # self.rgb_mean = torch.tensor([0.4488, 0.4371, 0.4040], requires_grad=False).view([1, 3, 1, 1]).cuda()
        self.register_buffer('rgb_mean', torch.tensor([0.4488, 0.4371, 0.4040]).view([1, 3, 1, 1]))

        # define head module
        self.head = nn.Sequential(wn(nn.Conv2d(config['n_colors'], n_feats, 3, padding=3 // 2)))

        # define body module
        self.body = nn.Sequential(*[IARB(n_feats, kernel_size, config['block_feats'], wn=wn, act=act) for _ in range(n_resblocks)])

        # define tail module
        out_feats = scale * scale * config['n_colors']
        self.tail = MUSM(config['n_colors'], scale, n_feats, wn)

        self.skip = nn.Sequential(wn(nn.Conv2d(config['n_colors'], out_feats, 3, padding=3 // 2)), nn.PixelShuffle(scale))

    def forward(self, x):
        x = (x - self.rgb_mean * 255) / 127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x = x * 127.5 + self.rgb_mean * 255
        return x


if __name__ == '__main__':
    import argparse
    from copy import deepcopy
    import yaml
    import time
    parser = argparse.ArgumentParser(description='Train the LROD model')
    parser.add_argument('--config_sr', type=str, default='../config/config_sr.yaml', help='configuration file for the SR subnetwork')
    parser.add_argument('--test_only', action='store_true', help='Is it just testing?')
    parser.add_argument('--test_model', type=str, default='', help='model to test')
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale')
    parser.add_argument('--workers', type=int, default=4, help='Loading data thread')
    parser.add_argument('--GPU', action='store_false', help='Whether to use GPU')
    parser.add_argument('--batch', type=int, default=16, help='Training batchSize')
    parser.add_argument('--epoch', type=int, default=1000, help='Training epochs')
    parser.add_argument('--out_path', type=str, default='./exp/output', help='dir to save output during training phase')
    parser.add_argument('--log_freq', type=int, default=10, help='frequency of save log during an epoch')
    parser.add_argument('--save_freq', type=int, default=10, help='frequency of save checkpoint during whole training phase')
    parser.add_argument('--resume', action='store_true', help='Is it continue train from last time')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint file need to be loaded')

    args = parser.parse_args()


    with open(args.config_sr, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = MODEL(config['Model'], args).cuda()

    a = torch.ones([1, 3, 320, 180]).cuda()
    t = time.time()
    b = model(a)
    print(time.time() - t)
    from thop import profile, clever_format

    flops, params = profile(model, inputs=a)
    macs, params = clever_format([flops, params], "%.3f")  # 格式化输出
    print('flops', macs)  # 计算量
    print('params:', params)  # 模型参数量

    pass

