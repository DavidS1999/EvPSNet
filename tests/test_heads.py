import mmcv
import torch

# old imports
# from mmdet.core import build_assigner, build_sampler
# from mmdet.models.anchor_heads import AnchorHead
# from mmdet.models.bbox_heads import BBoxHead

# new imports
from mmdet.registry import TASK_UTILS
from mmdet.models.dense_heads.anchor_head import AnchorHead
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead


def test_anchor_head_loss():
    """
    Tests anchor head loss when truth is empty and non-empty
    """
    self = AnchorHead(num_classes=4, in_channels=1)
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]

    cfg = mmcv.Config({
        'assigner': {
            'type': 'MaxIoUAssigner',
            'pos_iou_thr': 0.7,
            'neg_iou_thr': 0.3,
            'min_pos_iou': 0.3,
            'ignore_iof_thr': -1
        },
        'sampler': {
            'type': 'RandomSampler',
            'num': 256,
            'pos_fraction': 0.5,
            'neg_pos_ub': -1,
            'add_gt_as_proposals': False
        },
        'allowed_border': 0,
        'pos_weight': -1,
        'debug': False
    })

    # Anchor head expects a multiple levels of features per image
    feat = [
        torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
        for i in range(len(self.anchor_generators))
    ]
    cls_scores, bbox_preds = self.forward(feat)

    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                img_metas, cfg, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_cls_loss = sum(empty_gt_losses['loss_cls'])
    empty_box_loss = sum(empty_gt_losses['loss_bbox'])
    assert empty_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert empty_box_loss.item() == 0, (
        'there should be no box loss when there are no true boxes')

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                              img_metas, cfg, gt_bboxes_ignore)
    onegt_cls_loss = sum(one_gt_losses['loss_cls'])
    onegt_box_loss = sum(one_gt_losses['loss_bbox'])
    assert onegt_cls_loss.item() > 0, 'cls loss should be non-zero'
    assert onegt_box_loss.item() > 0, 'box loss should be non-zero'


def test_bbox_head_loss():
    """
    Tests bbox head loss when truth is empty and non-empty
    """
    self = BBoxHead(in_channels=8, roi_feat_size=3)

    num_imgs = 1
    feat = torch.rand(1, 1, 3, 3)

    # Dummy proposals
    proposal_list = [
        torch.Tensor([[23.6667, 23.8757, 228.6326, 153.8874]]),
    ]

    target_cfg = mmcv.Config({'pos_weight': 1})

    def _dummy_bbox_sampling(proposal_list, gt_bboxes, gt_labels):
        """
        Create sample results that can be passed to BBoxHead.get_target
        """
        assign_config = {
            'type': 'MaxIoUAssigner',
            'pos_iou_thr': 0.5,
            'neg_iou_thr': 0.5,
            'min_pos_iou': 0.5,
            'ignore_iof_thr': -1
        }
        sampler_config = {
            'type': 'RandomSampler',
            'num': 512,
            'pos_fraction': 0.25,
            'neg_pos_ub': -1,
            'add_gt_as_proposals': True
        }

        # old code
        # bbox_assigner = build_assigner(assign_config)
        # bbox_sampler = build_sampler(sampler_config)

        # new code
        bbox_assigner = TASK_UTILS.build(assign_config)
        bbox_sampler = TASK_UTILS.build(sampler_config)

        gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = bbox_assigner.assign(proposal_list[i],
                                                 gt_bboxes[i],
                                                 gt_bboxes_ignore[i],
                                                 gt_labels[i])
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=feat)
            sampling_results.append(sampling_result)
        return sampling_results

    # Test bbox loss when truth is empty
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    bbox_targets = self.get_target(sampling_results, gt_bboxes, gt_labels,
                                   target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 8 * 3 * 3)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox', 0) == 0, 'empty gt loss should be zero'

    # Test bbox loss when truth is non-empty
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]

    sampling_results = _dummy_bbox_sampling(proposal_list, gt_bboxes,
                                            gt_labels)

    bbox_targets = self.get_target(sampling_results, gt_bboxes, gt_labels,
                                   target_cfg)
    labels, label_weights, bbox_targets, bbox_weights = bbox_targets

    # Create dummy features "extracted" for each sampled bbox
    num_sampled = sum(len(res.bboxes) for res in sampling_results)
    dummy_feats = torch.rand(num_sampled, 8 * 3 * 3)
    cls_scores, bbox_preds = self.forward(dummy_feats)

    losses = self.loss(cls_scores, bbox_preds, labels, label_weights,
                       bbox_targets, bbox_weights)
    assert losses.get('loss_cls', 0) > 0, 'cls-loss should be non-zero'
    assert losses.get('loss_bbox', 0) > 0, 'box-loss should be non-zero'


def test_refine_boxes():
    """
    Mirrors the doctest in
    ``mmdet.models.bbox_heads.bbox_head.BBoxHead.refine_boxes`` but checks for
    multiple values of n_roi / n_img.
    """
    self = BBoxHead(reg_class_agnostic=True)

    test_settings = [

        # Corner case: less rois than images
        {
            'n_roi': 2,
            'n_img': 4,
            'rng': 34285940
        },

        # Corner case: no images
        {
            'n_roi': 0,
            'n_img': 0,
            'rng': 52925222
        },

        # Corner cases: few images / rois
        {
            'n_roi': 1,
            'n_img': 1,
            'rng': 1200281
        },
        {
            'n_roi': 2,
            'n_img': 1,
            'rng': 1200282
        },
        {
            'n_roi': 2,
            'n_img': 2,
            'rng': 1200283
        },
        {
            'n_roi': 1,
            'n_img': 2,
            'rng': 1200284
        },

        # Corner case: no rois few images
        {
            'n_roi': 0,
            'n_img': 1,
            'rng': 23955860
        },
        {
            'n_roi': 0,
            'n_img': 2,
            'rng': 25830516
        },

        # Corner case: no rois many images
        {
            'n_roi': 0,
            'n_img': 10,
            'rng': 671346
        },
        {
            'n_roi': 0,
            'n_img': 20,
            'rng': 699807
        },

        # Corner case: similar num rois and images
        {
            'n_roi': 20,
            'n_img': 20,
            'rng': 1200238
        },
        {
            'n_roi': 10,
            'n_img': 20,
            'rng': 1200238
        },
        {
            'n_roi': 5,
            'n_img': 5,
            'rng': 1200238
        },

        # ----------------------------------
        # Common case: more rois than images
        {
            'n_roi': 100,
            'n_img': 1,
            'rng': 337156
        },
        {
            'n_roi': 150,
            'n_img': 2,
            'rng': 275898
        },
        {
            'n_roi': 500,
            'n_img': 5,
            'rng': 4903221
        },
    ]

    for demokw in test_settings:
        try:
            n_roi = demokw['n_roi']
            n_img = demokw['n_img']
            rng = demokw['rng']

            print('Test refine_boxes case: {!r}'.format(demokw))
            tup = _demodata_refine_boxes(n_roi, n_img, rng=rng)
            rois, labels, bbox_preds, pos_is_gts, img_metas = tup
            bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
                                             pos_is_gts, img_metas)
            assert len(bboxes_list) == n_img
            assert sum(map(len, bboxes_list)) <= n_roi
            assert all(b.shape[1] == 4 for b in bboxes_list)
        except Exception:
            print('Test failed with demokw={!r}'.format(demokw))
            raise


def _demodata_refine_boxes(n_roi, n_img, rng=0):
    """
    Create random test data for the
    ``mmdet.models.bbox_heads.bbox_head.BBoxHead.refine_boxes`` method
    """
    import numpy as np

    # old imports
    # from mmdet.core.bbox.demodata import random_boxes
    # from mmdet.core.bbox.demodata import ensure_rng

    # functions from old mmdet
    def ensure_rng(rng=None):
        """
        Simple version of the ``kwarray.ensure_rng``

        Args:
            rng (int | numpy.random.RandomState | None):
                if None, then defaults to the global rng. Otherwise this can be an
                integer or a RandomState class
        Returns:
            (numpy.random.RandomState) : rng -
                a numpy random number generator

        References:
            https://gitlab.kitware.com/computer-vision/kwarray/blob/master/kwarray/util_random.py#L270
        """

        if rng is None:
            rng = np.random.mtrand._rand
        elif isinstance(rng, int):
            rng = np.random.RandomState(rng)
        else:
            rng = rng
        return rng


    def random_boxes(num=1, scale=1, rng=None):
        """
        Simple version of ``kwimage.Boxes.random``

        Returns:
            Tensor: shape (n, 4) in x1, y1, x2, y2 format.

        References:
            https://gitlab.kitware.com/computer-vision/kwimage/blob/master/kwimage/structs/boxes.py#L1390

        Example:
            >>> num = 3
            >>> scale = 512
            >>> rng = 0
            >>> boxes = random_boxes(num, scale, rng)
            >>> print(boxes)
            tensor([[280.9925, 278.9802, 308.6148, 366.1769],
                    [216.9113, 330.6978, 224.0446, 456.5878],
                    [405.3632, 196.3221, 493.3953, 270.7942]])
        """
        rng = ensure_rng(rng)

        tlbr = rng.rand(num, 4).astype(np.float32)

        tl_x = np.minimum(tlbr[:, 0], tlbr[:, 2])
        tl_y = np.minimum(tlbr[:, 1], tlbr[:, 3])
        br_x = np.maximum(tlbr[:, 0], tlbr[:, 2])
        br_y = np.maximum(tlbr[:, 1], tlbr[:, 3])

        tlbr[:, 0] = tl_x * scale
        tlbr[:, 1] = tl_y * scale
        tlbr[:, 2] = br_x * scale
        tlbr[:, 3] = br_y * scale

        boxes = torch.from_numpy(tlbr)
        return boxes

    try:
        import kwarray
    except ImportError:
        import pytest
        pytest.skip('kwarray is required for this test')
    scale = 512
    rng = ensure_rng(rng)
    img_metas = [{'img_shape': (scale, scale)} for _ in range(n_img)]
    # Create rois in the expected format
    roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
    if n_img == 0:
        assert n_roi == 0, 'cannot have any rois if there are no images'
        img_ids = torch.empty((0, ), dtype=torch.long)
        roi_boxes = torch.empty((0, 4), dtype=torch.float32)
    else:
        img_ids = rng.randint(0, n_img, (n_roi, ))
        img_ids = torch.from_numpy(img_ids)
    rois = torch.cat([img_ids[:, None].float(), roi_boxes], dim=1)
    # Create other args
    labels = rng.randint(0, 2, (n_roi, ))
    labels = torch.from_numpy(labels).long()
    bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
    # For each image, pretend random positive boxes are gts
    is_label_pos = (labels.numpy() > 0).astype(np.int)
    lbl_per_img = kwarray.group_items(is_label_pos, img_ids.numpy())
    pos_per_img = [sum(lbl_per_img.get(gid, [])) for gid in range(n_img)]
    # randomly generate with numpy then sort with torch
    _pos_is_gts = [
        rng.randint(0, 2, (npos, )).astype(np.uint8) for npos in pos_per_img
    ]
    pos_is_gts = [
        torch.from_numpy(p).sort(descending=True)[0] for p in _pos_is_gts
    ]
    return rois, labels, bbox_preds, pos_is_gts, img_metas
