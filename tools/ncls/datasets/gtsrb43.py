import sys
sys.path.append('./tools')
from utils import exec_cmd
import os
import glob
import argparse
import yaml
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--load_from_opt', action='store_true', default=False)
parser.add_argument('-mdr', '--meta_data_root', type=str, default='/datasets') # edit
parser.add_argument('-ds', '--dataset', type=str, default='gtsrb43') # edit
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
cat_root = f'{data_root}/GTSRB/Final_Training/Images'
file_name = f'GTSRB_Final_Training_Images.zip'
download_link = f'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/{file_name}'
exec_cmd(f'wget -P {data_root} {download_link}')

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

print('\ncat_root: ', cat_root)
class_dirs = sorted(glob.glob(f'{cat_root}/*'))
class_ls = sorted([class_dir.split('/')[-1] for class_dir in class_dirs])
print('\n class_ls:', class_ls)
'''
e.g.
class_ls: ['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009', '00010', '00011', '00012', '00013', '00014', '00015', '00016', '00017', '00018', '00019', '00020', '00021', '00022', '00023', '00024', '00025', '00026', '00027', '00028', '00029', '00030', '00031', '00032', '00033', '00034', '00035', '00036', '00037', '00038', '00039', '00040', '00041', '00042']
'''
print('\n len(class_ls):' , len(class_ls))
# 43
removed_classes = []

class_num = 0
for class_dir in class_dirs:
    class_num_str = str(class_num)
    class_name = class_dir.split('/')[-1]
    print('\nclass_name: ', class_name) # e.g. 00000
    if class_name not in removed_classes:
        while len(class_num_str) < 5: class_num_str = '0' + class_num_str
        img_paths = sorted(glob.glob(f'{class_dir}/*'))
        # print('\nimg_paths: ', img_paths)
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            # print('\nimg_path: ', img_path) # e.g. /datasets/gtsrb43/GTSRB/Final_Training/Images/00000/00000_00000.ppm
            if '.ppm' in img_path:
                img = Image.open(img_path)

                # print('\nimg_name: ', img_name) # e.g. image_0058.jpg
                img_id = img_name[:img_name.index('.')]
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
                img.save(img_path_DST)
                print(f'\n{img_path_DST} saved!')
        class_num += 1

# Clean other dirs
exec_cmd(f'rm -r {data_root}/GTSRB')
exec_cmd(f'rm -r {data_root}/{file_name}')

