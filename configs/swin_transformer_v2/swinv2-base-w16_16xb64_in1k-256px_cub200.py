_base_ = [
    '../_base_/models/swin_transformer_v2/base_256_cub200.py', # edit
    '../_base_/datasets/cub200_bs64_swin_256.py', # edit
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime_save_best.py'
]
model = dict(backbone=dict(window_size=[16, 16, 16, 8]))
#=================================
# Updated for the new dataset >>>
# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=200, # edit
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=256),
    dict(type='RandomResizedCrop', scale=256), # 224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=256),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=256), # 224),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=16,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/datasets/cub200', # edit
        ann_file='meta/train.txt', # edit
        data_prefix='train', # edit
        # split='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/datasets/cub200', # edit
        ann_file='meta/val.txt', # edit
        data_prefix='val', # edit
        # split='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))
# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
# Updated for the new dataset <<<
#=================================
