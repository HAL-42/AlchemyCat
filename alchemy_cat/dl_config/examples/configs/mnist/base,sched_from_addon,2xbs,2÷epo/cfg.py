# -- configs/mnist/base,sched_from_addon,2xlr,2÷bs/cfg.py --

from alchemy_cat.dl_config import Config

cfg = Config(caps='configs/mnist/base,sched_from_addon/cfg.py')

cfg.loader.ini.batch_size = 256

cfg.sched.epochs = 15
