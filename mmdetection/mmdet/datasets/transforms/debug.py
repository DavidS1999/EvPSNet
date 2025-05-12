from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
import torch

@TRANSFORMS.register_module()
class DebugBboxCheck(BaseTransform):
    def transform(self, results):
        if 'gt_bboxes' in results:
            bboxes = results['gt_bboxes']
            if hasattr(bboxes, 'tensor'):
                print("[DEBUG] BBoxes shape:", bboxes.tensor.shape)
            else:
                print("[DEBUG] BBoxes raw:", bboxes)
        else:
            print("[DEBUG] Keine gt_bboxes gefunden.")
        return results

@TRANSFORMS.register_module()
class ConvertBoxesToNumpy(BaseTransform):
    """Ensure gt_bboxes are NumPy arrays."""

    def transform(self, results):
        gt_bboxes = results.get('gt_bboxes', None)
        if gt_bboxes is not None:
            if hasattr(gt_bboxes, 'tensor'):  # BaseBoxes wie HorizontalBoxes
                results['gt_bboxes'] = gt_bboxes.tensor.cpu().numpy()
            elif isinstance(gt_bboxes, torch.Tensor):  # Falls es ein Tensor ist
                results['gt_bboxes'] = gt_bboxes.cpu().numpy()
            elif isinstance(gt_bboxes, list):
                results['gt_bboxes'] = np.array(gt_bboxes)
        return results