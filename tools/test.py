import argparse
import os

import torch
from torch.nn import Linear
import numpy as np
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.fileio import dump
from mmengine.dist import init_dist, get_dist_info
from mmengine.logging import HistoryBuffer
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmdet.registry import MODELS, DATASETS
from mmdet.evaluation import CocoMetric

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


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection test script')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--work-dir', type=str, help='Directory to save logs and results')
    parser.add_argument('--out', help='Output result file in pickle format (.pkl)')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='Evaluation metrics, e.g. "bbox", "segm", "panoptic"')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='Job launcher for distributed testing')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # Initialize distributed testing if needed
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.get('dist_params', {}))

    # Build model and load checkpoint
    model = MODELS.build(cfg.model)
    
    torch.serialization.add_safe_globals([HistoryBuffer, np.core.multiarray._reconstruct])

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    from mmengine.runner.checkpoint import _load_checkpoint_to_model
    _load_checkpoint_to_model(model, checkpoint, strict=False, logger=None)

    # load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    # Wrap model in appropriate parallel wrapper
    # TODO

    # if not distributed:
    #     model = DataParallel(model, device_ids=[0])
    # else:
    #     model = DistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False
    #     )

    # Build test dataset and dataloader

    dataset = cfg.test_dataloader['dataset']
    test_dataloader = cfg.test_dataloader


    # dataset = DATASETS.build(cfg.data.test)
    # test_dataloader = cfg.get('test_dataloader', None)
    # if test_dataloader is None:
    #     from mmengine.dataset import default_collate
    #     from torch.utils.data import DataLoader
    #     test_dataloader = DataLoader(
    #         dataset,
    #         batch_size=1,
    #         shuffle=False,
    #         collate_fn=default_collate,
    #         num_workers=cfg.data.get('workers_per_gpu', 2),
    #         pin_memory=True
    #     )

    evaluator = cfg.test_evaluator

    # Define evaluator if evaluation metrics are given
    # evaluator = None
    # if args.eval:
    #     evaluator = [CocoMetric(metric) for metric in args.eval]

    # Use overridden work_dir if provided
    work_dir = args.work_dir or cfg.get('work_dir') or './work_dirs/tmp_test'

    # Build MMEngine runner
    # runner = Runner.from_cfg(dict(
    #     model=model,
    #     work_dir=work_dir,
    #     test_dataloader=test_dataloader,
    #     test_evaluator=evaluator,
    #     test_cfg=dict()
    # ))
    cfg.work_dir = work_dir
    cfg.test_dataloader = test_dataloader
    cfg.test_evaluator = evaluator
    cfg.test_cfg = dict()

    runner = Runner.from_cfg(cfg)
    runner.model = model

    # Run test loop
    results = runner.test()

    # Save results if needed (only rank 0)
    rank, _ = get_dist_info()
    if rank == 0 and args.out:
        dump(results, args.out)
        print(f'\nwriting results to {args.out}')


if __name__ == '__main__':
    main()
