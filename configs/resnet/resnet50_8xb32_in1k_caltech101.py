_base_ = [
    '../_base_/models/resnet50_caltech101.py', # EDIT
    '../_base_/datasets/caltech101_bs32.py', # EDIT
    '../_base_/schedules/imagenet_bs256.py', 
    '../_base_/default_runtime_save_best.py'
]
