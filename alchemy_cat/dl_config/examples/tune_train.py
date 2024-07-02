# -- tune_train.py --
import argparse, json, os, subprocess, torch, sys
from alchemy_cat.dl_config import Config, Cfg2TuneRunner

parser = argparse.ArgumentParser(description='Tuning AlchemyCat MNIST Example')
parser.add_argument('-c', '--cfg2tune', type=str)
args = parser.parse_args()

# Set `pool_size` to GPU num, will run `pool_size` of configs in parallel
runner = Cfg2TuneRunner(args.cfg2tune, experiment_root='/tmp/experiment', pool_size=torch.cuda.device_count())

@runner.register_work_fn  # How to run config
def work(pkl_idx: int, cfg: Config, cfg_pkl: str, cfg_rslt_dir: str) -> ...:
    subprocess.run([sys.executable, 'train.py', '-c', cfg_pkl],
                   env=os.environ | {'CUDA_VISIBLE_DEVICE': f'pkl_idx % torch.cuda.device_count()'})

@runner.register_gather_metric_fn  # How to gather metric for summary
def gather_metric(cfg: Config, cfg_rslt_dir: str, run_rslt: ..., param_comb: dict[str, tuple[..., str]]) -> dict[str, ...]:
    return json.load(open(os.path.join(cfg_rslt_dir, 'eval.json')))

runner.tuning()
