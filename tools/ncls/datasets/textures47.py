import sys
sys.path.append('./tools')
from utils import exec_cmd
import os
import glob
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--load_from_opt', action='store_true', default=False)
parser.add_argument('-mdr', '--meta_data_root', type=str, default='/datasets') # edit
parser.add_argument('-ds', '--dataset', type=str, default='textures47') # edit
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
file_name = 'dtd-r1.0.1.tar.gz'
download_link = f'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/{file_name}'
exec_cmd(f'wget -P {data_root} {download_link}')

# Untar the file
exec_cmd(f'tar -xvf {data_root}/{file_name} -C {data_root}')


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

data_root_tmp = f'{data_root}/dtd'

class_dirs = sorted(glob.glob(f'{data_root_tmp}/images/*'))
class_ls = sorted([class_dir.split('/')[-1] for class_dir in class_dirs])
print('\n class_ls:', class_ls)
'''
e.g.
class_ls: ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged']
'''
removed_classes = []

# Construct dataset
dataset_types = ['train', 'val']
for dataset_type in dataset_types:
    label_file_paths = sorted(glob.glob(f'{data_root_tmp}/labels/{dataset_type}1.txt'))
    print('\nlabel_file_paths: ', label_file_paths)
    
    for path in label_file_paths:
        label_file = open(path, 'r')
        lines = label_file.readlines()
        for line in lines:
            line = line[:-1]
            print('\npath: ', path)
            print('\ndataset_type: ', dataset_type, ', line: ', line) # e.g. banded/banded_0005.jpg
            img_path_rel = line
            spl = img_path_rel.split('/')
            class_name = spl[0]
            # print('\nclass_name: ', class_name) # e.g. banded
            class_num = class_ls.index(class_name)
            class_num_str = str(class_num)
            while len(class_num_str) < 2: class_num_str = '0' + class_num_str
            # print('\nclass_num_str: ', class_num_str)
            
            img_name = spl[1]
            img_path = f'{data_root_tmp}/images/{class_name}/{img_name}'
            # print('\nimg_path: ', img_path) # e.g. /datasets/textures47/dtd/images/banded_0005.jpg

            img_id = img_name[:img_name.index('.')]
            # print('\nimg_id: ', img_id) # e.g. banded_0005

            img_file_DST = f'{class_num_str}_image_{img_id}.jpg'
            meta_write_line = f'{img_file_DST} {class_num}\n'
            
            # Write to the meta file
            if dataset_type == 'train':
                # Write to the meta file
                train_meta_file.write(meta_write_line)
                train_meta_file.flush(); print(f'{meta_write_line}')

                # Move the dataset directory
                img_path_DST = f'{train_dir}/{img_file_DST}'
            elif dataset_type == 'val':
                # Write to the meta file
                val_meta_file.write(meta_write_line)
                val_meta_file.flush(); print(f'{meta_write_line}')

                # Move the dataset directory
                img_path_DST = f'{val_dir}/{img_file_DST}'
            exec_cmd(f'mv {img_path} {img_path_DST}')

# Clean other dirs
exec_cmd(f'rm -r {data_root}/{file_name}')
exec_cmd(f'rm -r {data_root_tmp}')
