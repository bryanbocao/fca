# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileViT', arch='small'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100, # edit
        in_channels=640,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
