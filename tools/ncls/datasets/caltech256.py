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
parser.add_argument('-ds', '--dataset', type=str, default='caltech256') # edit
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
cat_root_name = '256_ObjectCategories'
file_name = f'{cat_root_name}.tar'
download_link = f'https://data.caltech.edu/records/nyy15-4j048/files/{file_name}'
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

class_dirs = sorted(glob.glob(f'{data_root}/{cat_root_name}/*'))
class_ls = sorted([class_dir.split('/')[-1] for class_dir in class_dirs])
print('\n class_ls:', class_ls)
'''
e.g.
class_ls: ['001.ak47', '002.american-flag', '003.backpack', '004.baseball-bat', '005.baseball-glove', '006.basketball-hoop', '007.bat'
'''
print('\n len(class_ls):' , len(class_ls))
# 257

removed_classes = []

class_num = 0
for class_dir in class_dirs:
    class_num_str = str(class_num)
    class_name = class_dir.split('/')[-1]
    print('\nclass_name: ', class_name) # e.g. 001.ak47
    
    if class_name not in removed_classes:
        while len(class_num_str) < 3: class_num_str = '0' + class_num_str
        img_paths = sorted(glob.glob(f'{class_dir}/*'))
        # print('\nimg_paths: ', img_paths)
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            print('\nimg_path: ', img_path) # e.g. /datasets/caltech101/101_ObjectCategories/BACKGROUND_Google/img_0467.jpg
            if '.jpg' in img_path:
                # print('\nimg_name: ', img_name) # e.g. image_0058.jpg
                img_id = img_name[img_name.index('_') + 1:img_name.index('.')]
                # print('\nimg_id: ', img_id)

                img_file_DST = f'{class_num_str}_image_{img_id}.jpg'
                meta_write_line = f'{img_file_DST} {class_num}\n'
                if int(img_id) % 5 == 0:
                    # val
                    # Write to the meta file
                    val_meta_file.write(meta_write_line)
                    val_meta_file.flush(); print(f'{meta_write_line}')

                    # Move the dataset directory
                    img_path_DST = f'{val_dir}/{img_file_DST}'
                else:
                    # train
                    # Write to the meta file
                    train_meta_file.write(meta_write_line)
                    train_meta_file.flush(); print(f'{meta_write_line}')

                    # Move the dataset directory
                    img_path_DST = f'{train_dir}/{img_file_DST}'
                exec_cmd(f'mv {img_path} {img_path_DST}')
        class_num += 1

# Clean other dirs
exec_cmd(f'rm -r {data_root}/{cat_root_name}')
exec_cmd(f'rm -r {data_root}/{file_name}')

