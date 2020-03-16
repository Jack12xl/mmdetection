cudnn_benchmark = True
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg,
        num_outs=5),
    bbox_head=dict(
        type='RetinaSepBNHead',
        num_classes=2,
        num_ins=5,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.45,
    nms=dict(type='nms', iou_thr=0.09),
    max_per_img=600)
# dataset settings
dataset_type = 'MatSTUDataset'
data_root = ''

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(640, 640)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

import os.path as osp
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=osp.join("data", "detection_data", "partb_train.pkl"),
        pipeline=train_pipeline,
        data_root=data_root
    ),
    val=dict(
        type=dataset_type,
        ann_file=osp.join("data", "detection_data", "partb_test.pkl"),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        pipeline=test_pipeline))
evaluation = dict(interval=25, metric='bbox')
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.08,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[30, 40])
checkpoint_config = dict(interval=25)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 100
device_ids = range(1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/stu_retinanet_crop640_r50_fpn_50e'
load_from = './checkpoints/retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
resume_from = None
workflow = [('train', 1)]
