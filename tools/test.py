import argparse
import os

import torch
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from mmengine.fileio import dump
from mmengine.dist import init_dist, get_dist_info
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmdet.registry import MODELS, DATASETS
from mmdet.evaluation import CocoMetric


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
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    # Wrap model in appropriate parallel wrapper
    if not distributed:
        model = DataParallel(model, device_ids=[0])
    else:
        model = DistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False
        )

    # Build test dataset and dataloader
    dataset = DATASETS.build(cfg.data.test)
    test_dataloader = cfg.get('test_dataloader', None)
    if test_dataloader is None:
        from mmengine.dataset import default_collate
        from torch.utils.data import DataLoader
        test_dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=default_collate,
            num_workers=cfg.data.get('workers_per_gpu', 2),
            pin_memory=True
        )

    # Define evaluator if evaluation metrics are given
    evaluator = None
    if args.eval:
        evaluator = [CocoMetric(metric) for metric in args.eval]

    # Use overridden work_dir if provided
    work_dir = args.work_dir or cfg.get('work_dir', './work_dirs/tmp_test')

    # Build MMEngine runner
    runner = Runner.from_cfg(dict(
        model=model,
        work_dir=work_dir,
        test_dataloader=test_dataloader,
        test_evaluator=evaluator,
        test_cfg=dict()
    ))

    # Run test loop
    results = runner.test()

    # Save results if needed (only rank 0)
    rank, _ = get_dist_info()
    if rank == 0 and args.out:
        dump(results, args.out)
        print(f'\nwriting results to {args.out}')


if __name__ == '__main__':
    main()
