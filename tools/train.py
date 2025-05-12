from __future__ import division
import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from torch.nn import Linear
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from mmdet.utils import register_all_modules
register_all_modules()

from mmengine.registry import MODELS as ENGINE_MODELS
from mmengine.registry import TASK_UTILS as ENGINE_TASK_UTILS
from mmengine.registry import HOOKS as ENGINE_HOOKS
from mmengine.registry import DATASETS as ENGINE_DATASETS
from mmengine.registry import TRANSFORMS as ENGINE_TRANSFORMS
from mmengine.registry import METRICS as ENGINE_METRICS

from mmdet.datasets.transforms import LoadAnnotations, RandomCrop, PackDetInputs, FixBoundingBoxes, DebugBboxCheck, ConvertBoxesToNumpy

from mmdet.datasets import CityscapesDataset

from mmdet.models import EfficientPS, TWOWAYFPN, RPNHead, CrossEntropyLoss, SmoothL1Loss, SingleRoIExtractor, ConvFCBBoxHead, EvidenceClassLoss, FCNSepMaskHead, EfficientPSSemanticHead
from mmdet.models.task_modules import DeltaXYWHBBoxCoder, AnchorGenerator, MaxIoUAssigner, BboxOverlaps2D, RandomSampler

from mmdet.models.hooks import HeadHook

from mmdet.evaluation import CocoPanopticMetric

ENGINE_METRICS.register_module(module=CocoPanopticMetric)

ENGINE_TRANSFORMS.register_module(module=LoadAnnotations, force=True)
ENGINE_TRANSFORMS.register_module(module=RandomCrop)
ENGINE_TRANSFORMS.register_module(module=PackDetInputs)
ENGINE_TRANSFORMS.register_module(module=FixBoundingBoxes)
ENGINE_TRANSFORMS.register_module(module=DebugBboxCheck)
ENGINE_TRANSFORMS.register_module(module=ConvertBoxesToNumpy)

ENGINE_DATASETS.register_module(module=CityscapesDataset)

ENGINE_HOOKS.register_module(module=HeadHook)

ENGINE_MODELS.register_module(module=EfficientPS)
ENGINE_MODELS.register_module(module=TWOWAYFPN)
ENGINE_MODELS.register_module(module=RPNHead)
ENGINE_MODELS.register_module(module=CrossEntropyLoss)
ENGINE_MODELS.register_module(module=SmoothL1Loss)
ENGINE_MODELS.register_module(module=SingleRoIExtractor)
ENGINE_MODELS.register_module(module=ConvFCBBoxHead)
ENGINE_MODELS.register_module(module=EvidenceClassLoss)
ENGINE_MODELS.register_module(module=FCNSepMaskHead)
ENGINE_MODELS.register_module(module=EfficientPSSemanticHead)

ENGINE_TASK_UTILS.register_module(module=DeltaXYWHBBoxCoder)
ENGINE_TASK_UTILS.register_module(module=AnchorGenerator)
ENGINE_TASK_UTILS.register_module(module=MaxIoUAssigner)
ENGINE_TASK_UTILS.register_module(module=BboxOverlaps2D)
ENGINE_TASK_UTILS.register_module(module=RandomSampler)

ENGINE_TASK_UTILS.register_module(module=Linear)

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
    # cfg.train_cfg.val = args.validate

    # Build runner from config and start training
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
