# -- configs/addons/linear_warm_cos_sched.py --
import torch.optim.lr_scheduler as sched

from alchemy_cat.dl_config import Config, DEP

cfg = Config()

cfg.epochs = 30

@cfg.set_DEP(priority=0)  # warm_epochs = 10% of total epochs
def warm_epochs(c: Config) -> int:
    return round(0.1 * c.epochs)

cfg.warm.cls = sched.LinearLR
cfg.warm.ini.total_iters = DEP(lambda c: c.warm_epochs, priority=1)
cfg.warm.ini.start_factor = 1e-5
cfg.warm.ini.end_factor = 1.

cfg.main.cls = sched.CosineAnnealingLR
cfg.main.ini.T_max = DEP(lambda c: c.epochs - c.warm.ini.total_iters,
                         priority=2)  # main_epochs = total_epochs - warm_epochs
