# -- configs/mnist/base,2xbs/cfg.py --

from alchemy_cat.dl_config import Config

cfg = Config(caps='configs/mnist/base/cfg.py')

cfg.loader.ini.batch_size = 128 * 2
