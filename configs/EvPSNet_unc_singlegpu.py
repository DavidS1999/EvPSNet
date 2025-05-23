cudnn_benchmark = True
randomness = dict(seed=None, deterministic=False)

# model settings
model = dict(
    type='EfficientPS',
    pretrained=True,
    backbone=dict(
        type='tf_efficientnet_b5',
        act_cfg = dict(type="Identity"),  
        norm_cfg=dict(type='SyncBN', requires_grad=True), # type='InPlaceABNSync', activation='leaky_relu', activation_param=0.01,
        style='pytorch'),
    neck=dict(
        type='TWOWAYFPN',
        in_channels=[40, 64, 176, 2048], #b0[24, 40, 112, 1280], #b4[32, 56, 160, 1792], #b5[40, 64, 176, 2048],
        out_channels=256,
        norm_cfg=dict(type='SyncBN', requires_grad=True), # type='InPlaceABNSync'
        act_cfg=None,
        num_outs=4),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='ConvFCBBoxHead',
        num_shared_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=9,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        norm_cfg=dict(type='SyncBN', activation='leaky_relu', activation_param=0.01, requires_grad=True), # type='InPlaceABNSync'
        reg_class_agnostic=False,
        loss_cls=dict(
            type='EvidenceClassLoss', use_sigmoid=False, loss_weight=1.0, max_epoch=60, coef=0.06),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='FCNSepMaskHead',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=9,
        norm_cfg=dict(type='SyncBN', requires_grad=True), # type='InPlaceABNSync'
        act_cfg=None,
        loss_mask=dict(
            type='EvidenceClassLoss', use_mask=True, loss_weight=1.0, max_epoch=60, coef=0.06)),
    semantic_head=dict(
        type='EfficientPSSemanticHead',
        in_channels=256,
        conv_out_channels=128,
        num_classes=19,
        ignore_label=255,
        loss_weight=1.0,
        coef = 0.06,
        max_epoch=60,
        use_unc= True,
        use_lovasz = True,
        norm_cfg=dict(type='SyncBN', requires_grad=True), # type='InPlaceABNSync'
        act_cfg=None),
    use_unc=True,
    out_dir_unc="./tmpDir",
    train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000, # old
        max_per_img=2000, # new
        nms_thr=0.7,
        min_bbox_size=0,
        nms=dict(type='nms', iou_threshold=0.7),
    ),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False)),
    test_cfg = dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000, # old
            max_per_img=1000, # new
            nms_thr=0.7,
            min_bbox_size=0,
            nms=dict(type='nms', iou_threshold=0.7)),
        rcnn=dict(
            score_thr=0.5,
            nms=dict(type='nms', iou_thr=0.5),
            max_per_img=100,
            mask_thr_binary=0.5),
        panoptic=dict(
            overlap_thr=0.5,
            min_stuff_area=2048))
)


# new train val test configs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=160, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# dataset settings
dataset_type = 'CityscapesDataset'
data_root = './data/cityscapes/'
img_norm_cfg = dict(
    mean=[106.433, 116.617, 119.559], std=[65.496, 67.6, 74.123], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg = True),
    dict(type='ConvertBoxesToNumpy'),
    # dict(type='Resize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1024, 512), (1280, 640), (1536, 768), (1792, 896), (2048, 1024)],
        keep_ratio=True
    ),
    dict(type='RandomCrop', crop_size=(1024, 2048)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
    # dict(type='DefaultFormatBundle'),
    #  dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True), # new
    dict(type='RandomFlip', prob=0.0), # new
    dict(type='PackDetInputs') # new
    # dict(
    #     type='MultiScaleFlipAug',
    #     scale=(2048, 1024),
    #     flip=False,
    #     transforms=[
    #         dict(type='Resize', keep_ratio=True),
    #         dict(type='RandomFlip'),
    #         dict(type='Normalize', **img_norm_cfg),
    #         dict(type='Pad', size_divisor=32),
    #         dict(type='ImageToTensor', keys=['img']),
    #         dict(type='Collect', keys=['img']),
    #     ])
]

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale=s, keep_ratio=True)
             for s in [(1024, 512), (2048, 1024)]],
            [dict(type='RandomFlip', prob=0.0),
             dict(type='RandomFlip', prob=1.0)],
            [dict(type='PackDetInputs')]
        ]
    )
]

#custom hook for epoch numbers
custom_hooks = [
    dict(type="HeadHook")
]

# old data block

# data = dict(
#     imgs_per_gpu=1,
#     workers_per_gpu=1,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/train.json',
#         img_prefix=data_root + 'train/',
#          seg_prefix=data_root + 'stuffthingmaps/train/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/val.json',
#         img_prefix=data_root + 'val/',
#         seg_prefix=data_root + 'stuffthingmaps/val/',
#         panoptic_gt=data_root + 'cityscapes_panoptic_val',  
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/val.json',
#         img_prefix=data_root + 'val/',
#         seg_prefix=data_root + 'stuffthingmaps/val/',
#         panoptic_gt=data_root + 'cityscapes_panoptic_val',
#         pipeline=test_pipeline))


# new data block

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        data_prefix=dict(
            img=data_root + 'train/',
            seg=data_root + 'stuffthingmaps/train/',
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        data_prefix=dict(
            img=data_root + 'val/',
            seg=data_root + 'stuffthingmaps/val/',
        ),
        panoptic_gt=data_root + 'cityscapes_panoptic_val',
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        data_prefix=dict(
            img=data_root + 'val/',
            seg=data_root + 'stuffthingmaps/val/',
        ),
        panoptic_gt=data_root + 'cityscapes_panoptic_val',
        pipeline=test_pipeline
    )
)

# old evaluation
# evaluation = dict(interval=10, metric=['panoptic'])


# new evaluation
val_evaluator = dict(type='CocoPanopticMetric',
                     seg_prefix  = data_root + 'stuffthingmaps/val/')
test_evaluator = dict(type='CocoPanopticMetric',
                      seg_prefix  = data_root + 'stuffthingmaps/val/')


# old optimizer
# optimizer = dict(type='SGD', lr=0.07, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.07, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)


# learning policy

# old lr block
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[120, 144])

# new lr
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=160, by_epoch=True, milestones=[120, 144], gamma=0.1)
]
auto_scale_lr = dict(enable=True, base_batch_size=8)

# checkpoint_config = dict(interval=10)
# yapf:disable

# old log
# log_config = dict(
#     interval=1,
#     hooks=[
#         dict(type='TextLoggerHook'),
#         dict(type='TensorboardLoggerHook')
#     ])

# new log
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    timer=dict(type='IterTimerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
)


# yapf:enable
# runtime settings
total_epochs = 160
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
