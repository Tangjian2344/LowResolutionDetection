from LROD.model import MODELSR_, MODELSR1_, MODELSR2_, MODELSR3_, MODELSR4_, MODELSR5_, MODELSR6_, MODELSR7_


def make_model(config, args):
    model = 'MODELSR' + args.number
    model = getattr(eval(model), 'MODEL')(config, args)
    return model
