# -- configs/tune/tune_bs_epoch/cfg.py --

from alchemy_cat.dl_config import Cfg2Tune, Param2Tune

cfg = Cfg2Tune(caps='configs/mnist/base,sched_from_addon/cfg.py')

cfg.loader.ini.batch_size = Param2Tune([128, 256, 512])

cfg.sched.epochs = Param2Tune([5, 15])
