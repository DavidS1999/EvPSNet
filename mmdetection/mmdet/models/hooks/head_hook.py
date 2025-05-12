from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class HeadHook(Hook):

    def before_train_epoch(self, runner):
        model = runner.model

        dataloader_len = len(runner.train_dataloader)
        epoch = runner.epoch

        if hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'loss_cls'):
            model.bbox_head.loss_cls.max_iter = dataloader_len
            model.bbox_head.loss_cls.epoch = epoch

        if hasattr(model, 'mask_head') and hasattr(model.mask_head, 'loss_mask'):
            model.mask_head.loss_mask.max_iter = dataloader_len
            model.mask_head.loss_mask.epoch = epoch

        if hasattr(model, 'semantic_head'):
            model.semantic_head.max_iter = dataloader_len
            model.semantic_head.epoch = epoch

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        model = runner.model
        iter = runner.iter

        if hasattr(model, 'bbox_head') and hasattr(model.bbox_head, 'loss_cls'):
            model.bbox_head.loss_cls.iter = iter

        if hasattr(model, 'mask_head') and hasattr(model.mask_head, 'loss_mask'):
            model.mask_head.loss_mask.iter = iter

        if hasattr(model, 'semantic_head'):
            model.semantic_head.iter = iter