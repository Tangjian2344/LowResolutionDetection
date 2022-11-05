from LROD.utility import *
import torch.nn as nn
import torch.optim as opt
import numpy as np
import torch.optim.lr_scheduler as lr_sch
import time
import torch


class Trainer:
    def __init__(self, args, loader_dic, model, logger, config=None, writer=None):
        self.config = config
        self.args = args
        self.logger = logger
        self.model = model
        self.writer = writer

        if args.test_only:
            self.loader_test = loader_dic
        else:
            self.t0 = time.time()
            self.start_epo = 1
            self.loader_train, self.loader_test = loader_dic['train'], loader_dic['test']

    def train(self):
        self.model.train()
        optimizer, lr_scheduler, myloss = self.prepare()

        name, loader_train = self.loader_train
        tot_iter = len(loader_train)
        log_step = tot_iter // self.args.log_freq
        save_step = self.args.epoch // self.args.save_freq

        flag = 'continue ' if self.args.resume else ''
        self.logger.info('{}training in {} dataset'.format(flag, name))

        for epo in range(self.start_epo, self.args.epoch + 1):
            cur_iter = 1
            for lr, hr, _, _ in loader_train:
                lr, hr = lr.cuda(), hr.cuda()
                sr = self.model(lr)
                loss = myloss(sr, hr)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if cur_iter % log_step == 0:
                    self.writer.add_scalar('loss', loss.item(), tot_iter * (epo - 1) + cur_iter)
                    self.logger.info('epoch:{:>4}/{} | iter:{:>4}/{} | loss:{:>8} | time:{}s | lr:{}'.
                                     format(epo, self.args.epoch, cur_iter, tot_iter, '%.4f' % loss, '%.4f' % self.timer(),  '%.6f' % lr_scheduler.get_last_lr()[0]))
                cur_iter += 1
                lr_scheduler.step()

            name, avg_psnr = self.test()
            self.writer.add_scalar('psnr-' + name, avg_psnr, epo)

            torch.save(self.model, self.args.out_path + '/model_latest.pth')
            if epo % save_step == 0:
                checkpoint = {'net': self.model,
                              'opt': optimizer.state_dict(),
                              'sch': lr_scheduler.state_dict(),
                              'epo': epo}
                torch.save(checkpoint, self.args.out_path + '/checkpoint_{}.pth'.format(epo))

    def test(self):
        if self.args.test_only:
            state_dict = torch.load(self.args.test_model)
            if isinstance(state_dict, nn.Module):
                state_dict = state_dict.state_dict()
            self.model.load_state_dict(state_dict)
            self.logger.info('test model has been loaded')
        self.model.eval()
        with torch.no_grad():
            for name, loader_test in self.loader_test.items():
                self.logger.info('Evaluating the PSNR-Y in {}'.format(name))
                PSNR = []
                for lr, hr, _, path in loader_test:
                    lr, hr = lr.cuda(), hr.cuda()
                    sr = quantize(self.model(lr))

                    import cv2
                    import matplotlib.pyplot as plt
                    save_dir = os.path.dirname(self.args.test_model)
                    save_dir = save_dir + '/benchmark' + '/SR/' + name + '/X' + str(self.args.scale)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    plt.imsave(save_dir + '/' + os.path.basename(path[0]), np.array(sr.squeeze(0).cpu()).transpose(1, 2, 0).astype(np.uint8))
                    # cv2.imwrite()

                    psnr_ = calc_psnr(sr, hr, scale=self.args.scale, benchmark=True)
                    PSNR.append(psnr_.item())
                avg_psnr = sum(PSNR) / len(PSNR)
                self.logger.info('The avrage PSNR-Y in {} is {} \n'.format(name, '%.4f' % avg_psnr))

            # from LROD.ssim import ssim
            # from skimage.metrics import structural_similarity as compare_ssim
            # import cv2
            # for name, loader_test in self.loader_test.items():
            #     self.logger.info('Evaluating the SSIM in {}'.format(name))
            #     SSIM = []
            #     for lr, hr, _ in loader_test:
            #         lr, hr = lr.cuda(), hr.cuda()
            #         sr = quantize(self.model(lr))
            #
            #         hr = np.array(hr.squeeze(0).cpu()).transpose(1, 2, 0)
            #         sr = np.array(sr.squeeze(0).cpu()).transpose(1, 2, 0)
            #         hr = hr[..., 0] * 0.299 + hr[..., 1] * 0.587 + hr[..., 2] * 0.114
            #         sr = sr[..., 0] * 0.299 + sr[..., 1] * 0.587 + sr[..., 2] * 0.114
            #         hr = hr[11:-11, 11: -11]
            #         sr = sr[11:-11, 11: -11]
            #
            #         ssim_ = compare_ssim(sr, hr, win_size=11, multichannel=True, sigma=1.5, data_range=255, use_sample_covariance=False, gaussian_weights=True)
            #         SSIM.append(ssim_)
            #     avg_ssim = sum(SSIM) / len(SSIM)
            #     self.logger.info('The avrage SSIM in {} is {} \n'.format(name, '%.4f' % avg_ssim))
        self.model.train()
        return name, avg_psnr

    def prepare(self):
        optimizer = getattr(opt, self.config['opt']['type'])(self.model.parameters(), **self.config['opt']['params'])
        lr_scheduler = getattr(lr_sch, self.config['lr_sch']['type'])(optimizer, **self.config['lr_sch']['params'])
        myloss = getattr(nn, self.config['loss']['type'])()
        if self.args.resume:
            ckeckpoint = torch.load(self.args.checkpoint)
            self.model.load_state_dict(ckeckpoint['net'].state_dict())
            optimizer.load_state_dict(ckeckpoint['opt'])
            lr_scheduler.load_state_dict(ckeckpoint['sch'])
            self.start_epo = ckeckpoint['epo'] + 1
            self.logger.info('checkpoint file has been loaded')
        return optimizer, lr_scheduler, myloss

    def timer(self):
        cur = time.time()
        ret = cur - self.t0
        self.t0 = cur
        return ret
