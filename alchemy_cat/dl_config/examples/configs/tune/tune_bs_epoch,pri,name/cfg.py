# -- configs/tune/tune_bs_epoch,pri,name/cfg.py --

from alchemy_cat.dl_config import Cfg2Tune, Param2Tune

cfg = Cfg2Tune(caps='configs/mnist/base,sched_from_addon/cfg.py')

cfg.loader.ini.batch_size = Param2Tune([128, 256, 512], optional_value_names=['1xbs', '2xbs', '4xbs'], priority=1)

cfg.sched.epochs = Param2Tune([5, 15], priority=0)
