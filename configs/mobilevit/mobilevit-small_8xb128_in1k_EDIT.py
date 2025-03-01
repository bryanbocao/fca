_base_ = [
    '../_base_/models/mobilevit/mobilevit_s_EDIT.py', # edit
    '../_base_/datasets/EDIT_bs32.py', # edit
    '../_base_/default_runtime_save_best.py',
    '../_base_/schedules/imagenet_bs256.py',
]

# no normalize for original implements
data_preprocessor = dict(
    # RGB format normalization parameters
    mean=[0, 0, 0],
    std=[255, 255, 255],
    # use bgr directly
    to_rgb=False,
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=288, edge='short'),
    dict(type='CenterCrop', crop_size=256),
    dict(type='PackInputs'),
]

train_dataloader = dict(batch_size=128)

val_dataloader = dict(
    batch_size=128,
    dataset=dict(pipeline=test_pipeline),
)
test_dataloader = val_dataloader



#=================================
# Updated for the new dataset >>>

# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=, # edit
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
        data_root='', # edit
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
        data_root='', # edit
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
