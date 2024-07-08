# -- configs/mnist/plain_usage,override_loader/cfg.py --

from alchemy_cat.dl_config import Config

cfg = Config(caps='configs/mnist/plain_usage/cfg.py')

cfg.loader.ini.override()
cfg.loader.ini.shuffle = False
cfg.loader.ini.drop_last = False
