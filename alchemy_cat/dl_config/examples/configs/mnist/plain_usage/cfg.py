# -- configs/mnist/plain_usage/cfg.py --

import torch.optim.lr_scheduler as sched
import torchvision.models as model
import torchvision.transforms as T
from torch import optim
from torchvision.datasets import MNIST

from alchemy_cat.dl_config import Config

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
cfg.model.ini.num_classes = 10

# -* Set optimizer.
cfg.opt.cls = optim.AdamW
cfg.opt.ini.lr = 0.01

# -* Set scheduler.
cfg.sched.epochs = 30
cfg.sched.warm_epochs = 3

cfg.sched.warm.cls = sched.LinearLR
cfg.sched.warm.ini.total_iters = 3
cfg.sched.warm.ini.start_factor = 1e-05
cfg.sched.warm.ini.end_factor = 1.0

cfg.sched.main.cls = sched.CosineAnnealingLR
cfg.sched.main.ini.T_max = 27

# -* Set logger.
cfg.log.save_interval = 6
