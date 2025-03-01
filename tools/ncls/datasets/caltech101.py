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
parser.add_argument('-ds', '--dataset', type=str, default='caltech101') # edit
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
download_link = 'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip'
exec_cmd(f'wget -P {data_root} {download_link}')

# Unzip the file
exec_cmd(f'unzip {data_root}/caltech-101.zip -d {data_root}')

# Untar the file
exec_cmd(f'tar -xf {data_root}/caltech-101/101_ObjectCategories.tar.gz -C {data_root}')

# Clean the dataset
exec_cmd(f'rm -r {data_root}/caltech-101')
exec_cmd(f'rm -r {data_root}/caltech-101.zip')
exec_cmd(f'rm -r {data_root}/__MACOSX')

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

class_dirs = sorted(glob.glob(f'{data_root}/101_ObjectCategories/*'))
class_ls = sorted([class_dir.split('/')[-1] for class_dir in class_dirs])
print('\n class_ls:', class_ls)
'''
e.g.
class_ls: ['BACKGROUND_Google', 'Faces', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']
'''
print('\n len(class_ls):' , len(class_ls))
# 102
removed_classes = ['BACKGROUND_Google']

class_num = 0
for class_dir in class_dirs:
    class_num_str = str(class_num)
    class_name = class_dir.split('/')[-1]
    print('\nclass_name: ', class_name) # e.g. yin_yang
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
exec_cmd(f'rm -r {data_root}/101_ObjectCategories')

