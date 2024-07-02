# -- configs/patches/cifar10.py --

import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from alchemy_cat.dl_config import Config

cfg = Config()

cfg.dt.set_whole()
cfg.dt.cls = CIFAR10
cfg.dt.ini.root = '/tmp/data'
cfg.dt.ini.transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
