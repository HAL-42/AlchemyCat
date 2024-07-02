# -- configs/mnist/base/cfg.py --

import torch.optim.lr_scheduler as sched
import torchvision.models as model
import torchvision.transforms as T
from torch import optim
from torchvision.datasets import MNIST

from alchemy_cat.dl_config import Config, DEP

cfg = Config()

cfg.rand_seed = 0

# -* Set datasets.
cfg.dt.cls = MNIST
cfg.dt.ini.root = '/tmp/data'
cfg.dt.ini.transform = T.Compose([T.Grayscale(3), T.ToTensor(), T.Normalize((0.1307,), (0.3081,)),])

# -* Set data loader.
cfg.loader.ini.batch_size = 128
cfg.loader.ini.num_workers = 2

# -* Set model.
cfg.model.cls = model.resnet18
cfg.model.ini.num_classes = DEP(lambda c: len(c.dt.cls.classes))

# -* Set optimizer.
cfg.opt.cls = optim.AdamW
cfg.opt.ini.lr = DEP(lambda c: c.loader.ini.batch_size // 128 * 0.01)  # Linear scaling rule.

# -* Set scheduler.
cfg.sched.epochs = 30
@cfg.sched.set_DEP(priority=0)  # warm_epochs = 10% of total epochs
def warm_epochs(c: Config) -> int:
    return round(0.1 * c.sched.epochs)

cfg.sched.warm.cls = sched.LinearLR
cfg.sched.warm.ini.total_iters = DEP(lambda c: c.sched.warm_epochs, priority=1)
cfg.sched.warm.ini.start_factor = 1e-5
cfg.sched.warm.ini.end_factor = 1.

cfg.sched.main.cls = sched.CosineAnnealingLR
cfg.sched.main.ini.T_max = DEP(lambda c: c.sched.epochs - c.sched.warm.ini.total_iters,
                               priority=2)  # main_epochs = total_epochs - warm_epochs

# -* Set logger.
cfg.log.save_interval = DEP(lambda c: c.sched.epochs // 5, priority=1)  # Save model at every 20% of total epochs.
