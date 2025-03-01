_base_ = [
    '../_base_/models/resnet50_cifar100.py', # edit
    '../_base_/datasets/cifar100_bs32.py', # edit
    '../_base_/schedules/imagenet_bs256.py', 
    '../_base_/default_runtime_save_best.py'
]
