# Refers to https://pytorch.org/blog/ml-models-torchvision-v0.9/#classification
_base_ = [
    '../_base_/models/mobilenet_v3/mobilenet_v3_small_cub.py', # edit
    '../_base_/datasets/cub_bs128_mbv3.py', # edit
    '../_base_/default_runtime_save_best.py',
]
# schedule settings
optim_wrapper = dict(
    optimizer=dict(
        type='RMSprop',
        lr=0.064,
        alpha=0.9,
        momentum=0.9,
        eps=0.0316,
        weight_decay=1e-5))
param_scheduler = dict(type='StepLR', by_epoch=True, step_size=2, gamma=0.973)
train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=1)
val_cfg = dict()
test_cfg = dict()
# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (8 GPUs) x (128 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
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
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/datasets/cub', # edit
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
        data_root='/datasets/cub', # edit
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
