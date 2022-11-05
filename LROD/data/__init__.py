import os
from LROD.data import srdata
from torch.utils.data import DataLoader


def make_dataloader(config, args):
    loader_dic = {}
    if not args.test_only:
        loader_dic = {'train': (config['name'], DataLoader(srdata.SRData(config, args),
                                                           shuffle=True,
                                                           batch_size=args.batch,
                                                           pin_memory=args.GPU,
                                                           num_workers=args.workers)),
                      'test': {config['val_set']['name']: DataLoader(Benchmark(config['val_set'], args, test_only=not args.test_only),
                                                                     shuffle=False,
                                                                     batch_size=1,
                                                                     pin_memory=args.GPU,
                                                                     num_workers=args.workers)}}
    else:
        for name in config['name']:
            config['name'] = name
            loader = DataLoader(Benchmark(config, args, test_only=args.test_only),
                                shuffle=False,
                                batch_size=1,
                                pin_memory=args.GPU,
                                num_workers=args.workers)
            loader_dic[name] = loader
    return loader_dic


class Benchmark(srdata.SRData):
    def _set_filesystem(self, dir_data):
        self.dir_hr = os.path.join(dir_data, 'HR', self.name, 'X{}'.format(self.scale))
        self.dir_lr = os.path.join(dir_data, 'LR_BI', self.name, 'X{}'.format(self.scale))

    def _scan(self):
        names_hr, names_lr = sorted(os.listdir(self.dir_hr)), sorted(os.listdir(self.dir_lr))
        names_hr = [os.path.join(self.dir_hr, name_hr) for name_hr in names_hr]
        names_lr = [os.path.join(self.dir_lr, name_lr) for name_lr in names_lr]
        return names_hr, names_lr, len(names_hr)
