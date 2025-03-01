_base_ = [
    '../_base_/models/convnext_v2/base_cub.py', # edit
    '../_base_/datasets/cub_bs64_swin_224.py', # edit
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime_save_best.py',
]
# dataset setting
train_dataloader = dict(batch_size=32)
# schedule setting
optim_wrapper = dict(
    optimizer=dict(lr=2.5e-3),
    clip_grad=None,
)
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=20)
]
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]
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
        data_root='/datasets/cub', #='/datasets/imagenet', # edit
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
        data_root='/datasets/cub', #='/datasets/imagenet', # edit
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
