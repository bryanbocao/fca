# https://www.tensorflow.org/datasets/overview

import sys
sys.path.append('./tools')
from utils import exec_cmd
import os
import glob
import argparse
import yaml

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', type=str, default='~/tensorflow_datasets/')
parser.add_argument('-o', '--load_from_opt', action='store_true', default=False)
parser.add_argument('-mdr', '--meta_data_root', type=str, default='/datasets') # edit
parser.add_argument('-ds', '--dataset', type=str, default='quickdraw345') # edit
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

exec_cmd('pip install tensorflow-datasets')


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

# Note it only contains 'train' split
ds = tfds.load('quickdraw_bitmap', split='train', as_supervised=True, shuffle_files=False, data_dir=args.data_dir)

print('\nlen(ds): ', len(ds)) # e.g. 50426266

# Build your input pipeline
ds = ds.take(len(ds))
for i, (image, label) in enumerate(tfds.as_numpy(ds)):
    print(image.shape, label)
    print(type(image), type(label), label)
    
    img_id = str(i)
    while len(img_id) < 8: img_id = '0' + img_id

    class_num = label
    class_num_str = str(class_num)
    while len(class_num_str) < 3: class_num_str = '0' + class_num_str

    img_file_DST = f'{class_num_str}_image_{img_id}.jpg'
    meta_write_line = f'{img_file_DST} {class_num}\n'

    # Write to the meta file
    if i % 5 == 0:
        # Write to the meta file
        val_meta_file.write(meta_write_line)
        val_meta_file.flush(); print(f'{meta_write_line}')

        img_path_DST = f'{val_dir}/{img_file_DST}'
    else:
        # Write to the meta file
        train_meta_file.write(meta_write_line)
        train_meta_file.flush(); print(f'{meta_write_line}')

        img_path_DST = f'{train_dir}/{img_file_DST}'

    img = Image.fromarray(np.squeeze(image), mode='L')
    img.save(img_path_DST)
    print(f'\n{img_path_DST} saved!')
        

