from __future__ import division
import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dist import init_dist
from mmengine.utils import mkdir_or_exist

# from mmdet import __version__
# from mmdet.apis import set_random_seed, train_detector
# from mmdet.datasets import build_dataset
# from mmdet.models import build_detector
# from mmdet.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    # parser.add_argument(
    #     '--gpus',
    #     type=int,
    #     default=1,
    #     help='number of gpus to use '
    #     '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument(
    #     '--autoscale-lr',
    #     action='store_true',
    #     help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    mkdir_or_exist(os.path.abspath(cfg.work_dir))

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.get('dist_params', {}))

    # set random seeds
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.deterministic = args.deterministic
    
    # optional
    cfg.train_cfg.val = args.validate

    # Build runner from config and start training
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
