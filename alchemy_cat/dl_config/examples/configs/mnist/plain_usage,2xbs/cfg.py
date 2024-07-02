# -- configs/mnist/plain_usage,2xbs/cfg.py --

from alchemy_cat.dl_config import Config

cfg = Config(caps='configs/mnist/plain_usage/cfg.py')

cfg.loader.ini.batch_size = 128 * 2

cfg.opt.ini.lr = 0.01 * 2
