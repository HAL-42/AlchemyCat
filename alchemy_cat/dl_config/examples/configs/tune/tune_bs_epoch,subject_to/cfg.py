# -- configs/tune/tune_bs_epoch,subject_to/cfg.py --

from alchemy_cat.dl_config import Cfg2Tune, Param2Tune

cfg = Cfg2Tune(caps='configs/mnist/base,sched_from_addon/cfg.py')

cfg.loader.ini.batch_size = Param2Tune([128, 256, 512])

cfg.sched.epochs = Param2Tune([5, 15],
                              subject_to=lambda cur_val: cur_val * cfg.loader.ini.batch_size.cur_val <= 15 * 128)
