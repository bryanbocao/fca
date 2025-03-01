import sys
sys.path.append('./tools')
from utils import exec_cmd
import os
import glob
import argparse
import yaml
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--load_from_opt', action='store_true', default=False)
parser.add_argument('-mdr', '--meta_data_root', type=str, default='/datasets') # edit
parser.add_argument('-ds', '--dataset', type=str, default='cifar100') # edit
parser.add_argument('-mdyfp', '--meta_data_yaml_file_path', type=str, default='./tools/ncls/datasets/ds.yaml') # edut
args = parser.parse_args()

if args.load_from_opt:
    data_root = f'{args.meta_data_root}/{args.dataset}'
else:
    with open(args.meta_data_yaml_file_path, 'r') as f:
        data = yaml.safe_load(f)
        data_root = f'{data["meta_data_root"]}/{args.dataset}'
print('\ndata_root: ', data_root)

if not os.path.exists(data_root):
    os.makedirs(data_root)

# Download the dataset
# https://github.com/knjcode/cifar2png
exec_cmd('pip install cifar2png')

tmp_dir = f'{data_root}/tmp'

exec_cmd(f'cifar2png cifar100 {tmp_dir}')

# Untar the file
exec_cmd(f'tar -xvf {tmp_dir}/cifar-100-python.tar.gz -C {data_root}')

train_dir = f'{data_root}/train'
exec_cmd(f'mkdir {train_dir}')

val_dir = f'{data_root}/val'
exec_cmd(f'mkdir {val_dir}')

meta_dir = f'{data_root}/meta'
exec_cmd(f'mkdir {meta_dir}')

train_meta_file_path = f'{meta_dir}/train.txt'
exec_cmd(f'touch {train_meta_file_path}')
train_meta_file = open(train_meta_file_path, 'a')

val_meta_file_path = f'{meta_dir}/val.txt'
exec_cmd(f'touch {val_meta_file_path}')
val_meta_file = open(val_meta_file_path, 'a')

dataset_types = ['train', 'test']
class_dirs = defaultdict()
for dataset_type in dataset_types:
    class_dirs[dataset_type] = sorted(glob.glob(f'{data_root}/tmp/{dataset_type}/*'))
class_ls = sorted([class_dir.split('/')[-1] for class_dir in class_dirs[dataset_type]])
print('\n class_ls:', class_ls)
'''
e.g.
class_ls: ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
'''
removed_classes = []

for dataset_type in dataset_types:
    class_num = 0
    for class_dir in class_dirs[dataset_type]:
        class_num_str = str(class_num)
        class_name = class_dir.split('/')[-1]
        print('\nclass_name: ', class_name) # e.g.
        if class_name not in removed_classes:
            while len(class_num_str) < 3: class_num_str = '0' + class_num_str
            img_paths = sorted(glob.glob(f'{class_dir}/*'))
            # print('\nimg_paths: ', img_paths)
            for img_path in img_paths:
                img_name = img_path.split('/')[-1]
                # print('\nimg_path: ', img_path) # e.g. /datasets/cifar100/tmp/test/wolf/0030.png
                if '.png' in img_path:
                    # print('\nimg_name: ', img_name) # e.g. wolf
                    img_id = img_name[:img_name.index('.')]
                    # print('\nimg_id: ', img_id)

                    img_file_DST = f'{class_num_str}_image_{img_id}.png'
                    meta_write_line = f'{img_file_DST} {class_num}\n'
                    if dataset_type == 'train':
                        # Write to the meta file
                        train_meta_file.write(meta_write_line)
                        train_meta_file.flush(); print(f'{meta_write_line}')
                        
                        # Move the dataset directory
                        img_path_DST = f'{train_dir}/{img_file_DST}'
                        print('\nimg_path_DST: ', img_path_DST)

                    else:
                        # Write to the meta file
                        val_meta_file.write(meta_write_line)
                        val_meta_file.flush(); print(f'{meta_write_line}')

                        # Move the dataset directory
                        img_path_DST = f'{val_dir}/{img_file_DST}'

                    exec_cmd(f'mv {img_path} {img_path_DST}')
            class_num += 1

# Clean other dirs
exec_cmd(f'rm -r {data_root}/tmp')

