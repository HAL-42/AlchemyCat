# -- train.py --
import argparse
import json

import torch
import torch.nn.functional as F
from rich.progress import track
from torch.optim.lr_scheduler import SequentialLR

from alchemy_cat.dl_config import load_config
from utils import eval_model

parser = argparse.ArgumentParser(description='AlchemyCat MNIST Example')
parser.add_argument('-c', '--config', type=str, default='configs/mnist/base,sched_from_addon,2xbs,2÷epo/cfg.py')
args = parser.parse_args()

# Folder 'experiment/mnist/base' will be auto created by `load` and assigned to `cfg.rslt_dir`
cfg = load_config(args.config, experiments_root='/tmp/experiment', config_root='configs')
print(cfg)

torch.manual_seed(cfg.rand_seed)  # Use `cfg` to set random seed

dataset = cfg.dt.cls(**cfg.dt.ini)  # Use `cfg` to set dataset type and its initial parameters

# Use `cfg` to set changeable parameters of loader,
# other fixed parameter like `shuffle` is set in main code
loader = torch.utils.data.DataLoader(dataset, shuffle=True, **cfg.loader.ini)

model = cfg.model.cls(**cfg.model.ini).train().to('cuda')  # Use `cfg` to set model

# Use `cfg` to set optimizer, and get `model.parameters()` in run time
opt = cfg.opt.cls(model.parameters(), **cfg.opt.ini, weight_decay=0.)

# Use `cfg` to set warm and main scheduler, and `SequentialLR` to combine them
warm_sched = cfg.sched.warm.cls(opt, **cfg.sched.warm.ini)
main_sched = cfg.sched.main.cls(opt, **cfg.sched.main.ini)
sched = SequentialLR(opt, [warm_sched, main_sched], [cfg.sched.warm_epochs])

for epoch in range(1, cfg.sched.epochs + 1):  # train `cfg.sched.epochs` epochs
    for data, target in track(loader, description=f"Epoch {epoch}/{cfg.sched.epochs}"):
        F.cross_entropy(model(data.to('cuda')), target.to('cuda')).backward()
        opt.step()
        opt.zero_grad()

    sched.step()

    # If cfg.log is defined, save model to `cfg.rslt_dir` at every `cfg.log.save_interval`
    if cfg.log and epoch % cfg.log.save_interval == 0:
        torch.save(model.state_dict(), f"{cfg.rslt_dir}/model_{epoch}.pth")

    eval_model(model)

if cfg.log:
    eval_ret = eval_model(model)
    with open(f"{cfg.rslt_dir}/eval.json", 'w') as json_f:
        json.dump(eval_ret, json_f)
