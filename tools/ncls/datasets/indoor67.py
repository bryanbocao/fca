import sys
sys.path.append('./tools')
from utils import exec_cmd
import os
import glob
import argparse
import yaml
import json

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--load_from_opt', action='store_true', default=False)
parser.add_argument('-mdr', '--meta_data_root', type=str, default='/datasets') # edit
parser.add_argument('-ds', '--dataset', type=str, default='indoor67') # edit
parser.add_argument('-mdyfp', '--meta_data_yaml_file_path', type=str, default='./tools/ncls/datasets/ds.yaml') # edut
parser.add_argument('-kjfp', '--kaggle_json_file_path', type=str, default='./tools/ncls/datasets/kaggle.json') # edit
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

# Set env vars
exec_cmd('pip install kaggle')
with open(args.kaggle_json_file_path, 'r') as f:
    data_json = json.load(f)
os.environ['KAGGLE_USERNAME'] = data_json['username']
os.environ['KAGGLE_KEY'] = data_json['key']
# print(os.environ['KAGGLE_USERNAME'])
# print(os.environ['KAGGLE_KEY'])

# Download the dataset
exec_cmd(f'kaggle datasets download -d itsahmad/indoor-scenes-cvpr-2019 -p {data_root}')
file_name = 'indoor-scenes-cvpr-2019.zip'

# Unzip the file
exec_cmd(f'unzip {data_root}/{file_name} -d {data_root}')
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

data_root_tmp = f'{data_root}/indoorCVPR_09'

class_dirs = sorted(glob.glob(f'{data_root_tmp}/Images/*'))
class_ls = sorted([class_dir.split('/')[-1] for class_dir in class_dirs])
print('\n class_ls:', class_ls)
'''
e.g.
class_ls: ['airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar', 'bathroom', 'bedroom', 'bookstore', 'bowling', 'buffet', 'casino', 'children_room', 'church_inside', 'classroom', 'cloister', 'closet', 'clothingstore', 'computerroom', 'concert_hall', 'corridor', 'deli', 'dentaloffice', 'dining_room', 'elevator', 'fastfood_restaurant', 'florist', 'gameroom', 'garage', 'greenhouse', 'grocerystore', 'gym', 'hairsalon', 'hospitalroom', 'inside_bus', 'inside_subway', 'jewelleryshop', 'kindergarden', 'kitchen', 'laboratorywet', 'laundromat', 'library', 'livingroom', 'lobby', 'locker_room', 'mall', 'meeting_room', 'movietheater', 'museum', 'nursery', 'office', 'operating_room', 'pantry', 'poolinside', 'prisoncell', 'restaurant', 'restaurant_kitchen', 'shoeshop', 'stairscase', 'studiomusic', 'subway', 'toystore', 'trainstation', 'tv_studio', 'videostore', 'waitingroom', 'warehouse', 'winecellar']
'''
print('\n len(class_ls): ', len(class_ls)) # e.g. len(class_ls):  67
removed_classes = []

# Construct dataset
dataset_types = ['train', 'val']
dataset_type_to_label_file_paths = {'train': 'TrainImages.txt', 'val': 'TestImages.txt'}
for dataset_type in dataset_types:
    label_file_path = f'{data_root}/{dataset_type_to_label_file_paths[dataset_type]}'
    print('\nlabel_file_path: ', label_file_path)
    
    # for path in label_file_path:
    path = label_file_path
    label_file = open(path, 'r')
    lines = label_file.readlines()
    for line_i, line in enumerate(lines):
        if '.jpg' in line and 'gif' not in line:
            if line_i < len(lines) - 1: line = line[:-1]
            img_path_rel = line
            spl = img_path_rel.split('/')
            class_name = spl[0]
            print('\nclass_name: ', class_name) # e.g. gameroom
            class_num = class_ls.index(class_name)
            class_num_str = str(class_num)
            while len(class_num_str) < 2: class_num_str = '0' + class_num_str
            print('\nclass_num_str: ', class_num_str)
            
            img_name = spl[1]
            edited = False
            if ' ' in img_name: # the original img_name contains ' '
                img_path = f'\"{data_root_tmp}/Images/{class_name}/{img_name}\"'
                img_name = img_name.replace(' ', '_') # to be used for img_path_DST
                edited = True
            if "'" in img_name:
                img_path = f'\"{data_root_tmp}/Images/{class_name}/{img_name}\"'
                img_name = img_name.replace("'", '_') # to be used for img_path_DST
                edited = True
            if not edited:
                img_path = f'{data_root_tmp}/Images/{class_name}/{img_name}'
            print('\nimg_path: ', img_path) # e.g. /datasets/indoor67/indoorCVPR_09/Images/gameroom/bt_132294gameroom2.jpg

            img_id = img_name[:img_name.index('.')]
            print('\nimg_id: ', img_id) # e.g. bt_132294gameroom2

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
            print(f'mv {img_path} {img_path_DST}')
            exec_cmd(f'mv {img_path} {img_path_DST}')

# Clean other dirs
exec_cmd(f'rm -r {data_root}/{file_name}')
exec_cmd(f'rm -r {data_root_tmp}')
exec_cmd(f'rm -r {data_root}/indoorCVPR_09annotations')
exec_cmd(f'rm -r {data_root}/TrainImages.txt')
exec_cmd(f'rm -r {data_root}/TestImages.txt')
