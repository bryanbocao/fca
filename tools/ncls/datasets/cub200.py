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
parser.add_argument('-ds', '--dataset', type=str, default='cub200') # edit
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
file_name = 'CUB_200_2011.tgz'
download_link = f'https://data.caltech.edu/records/65de6-vp158/files/{file_name}'
exec_cmd(f'wget -P {data_root} {download_link}')

# Untar the file
exec_cmd(f'tar -xzf {data_root}/{file_name} -C {data_root}')

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

data_root_tmp = f'{data_root}/CUB_200_2011'
# dataset_types = ['train', 'test']
# class_dirs = defaultdict()
# for dataset_type in dataset_types:
class_dirs = sorted(glob.glob(f'{data_root_tmp}/images/*'))
class_ls = sorted([class_dir.split('/')[-1] for class_dir in class_dirs])
print('\n class_ls:', class_ls)
'''
e.g.
class_ls: ['001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross', ..., '200.Common_Yellowthroat']
'''
removed_classes = []

images_txt_file = open(f'{data_root_tmp}/images.txt', 'r')
train_test_split_txt_file = open(f'{data_root_tmp}/train_test_split.txt', 'r')
images_txt_file_lines = images_txt_file.readlines()
train_test_split_txt_file_lines = train_test_split_txt_file.readlines()
print('\nlen(images_txt_file_lines): ', len(images_txt_file_lines)) # e.g. 11788
print('\nlen(train_test_split_txt_file_lines): ', len(train_test_split_txt_file_lines)) # e.g. 11788

class_num = 0
# for class_dir in class_dirs:
for line_i, line in enumerate(images_txt_file_lines):
    # images.txt
    lsp = line.split(' ')
    img_id, img_path_rel = lsp[0], lsp[1][:-1]
    lsp_1 = lsp[1].split('/')
    class_num = int(lsp_1[0].split('.')[0]) - 1 # Start with 0
    class_num_str = str(class_num)

    class_name = lsp_1[0]
    print('\nclass_name: ', class_name) # e.g. 001.Black_footed_Albatross
    
    # train_test_split.txt
    line_ = train_test_split_txt_file_lines[line_i]
    lsp_ = line_.split(' ')
    img_id_ = lsp_[0]
    assert img_id == img_id_
    train_label = True if int(lsp_[1]) == 1 else False

    if class_name not in removed_classes:
        while len(class_num_str) < 3: class_num_str = '0' + class_num_str
        img_path = f'{data_root_tmp}/images/{img_path_rel}'

        img_name = img_path.split('/')[-1]
        if '.jpg' in img_path:
            img_id = img_name[:img_name.index('.')]

            img_file_DST = f'{class_num_str}_image_{img_id}.jpg'
            meta_write_line = f'{img_file_DST} {class_num}\n'

            if train_label:
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
exec_cmd(f'rm -r {data_root}/CUB_200_2011')
exec_cmd(f'rm -r {data_root}/{file_name}')
exec_cmd(f'rm -r {data_root}/attributes.txt')

