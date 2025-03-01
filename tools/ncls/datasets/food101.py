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
parser.add_argument('-ds', '--dataset', type=str, default='food101') # edit
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
# file_name = 'food101_raw.tar.gz'
# download_link = 'http://data.vision.ee.ethz.ch/cvl/{file_name}' # unable to download
# download_link = f'https://huggingface.co/datasets/renumics/food101-enriched/resolve/main/data/{file_name}'
for i in range(8):
    download_link = f'https://huggingface.co/datasets/food101/resolve/main/data/train-0000{i}-of-00008.parquet'
    exec_cmd(f'wget -P {data_root} {download_link}')
for i in range(3):
    download_link = f'https://huggingface.co/datasets/food101/resolve/main/data/validation-0000{i}-of-00003.parquet'
    exec_cmd(f'wget -P {data_root} {download_link}')

# Prepare DST folders
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

class_ls = list(range(101))

# Read data
exec_cmd('pip install Pillow')
from PIL import Image
from io import BytesIO
def bytes_to_image(bytes):
    byte_stream = BytesIO(bytes)
    image = Image.open(byte_stream)
    return image

exec_cmd('pip install panda pyarrow fastparquet')
import pandas as pd

# Construct train set
train_labels = []
for i in range(8):
    df = pd.read_parquet(f'{data_root}/train-0000{i}-of-00008.parquet')
    sh = df.shape
    for j in range(sh[0]):
        class_num = df.iloc[j][1]
        train_labels.append(class_num)
        class_num_str = str(class_num)
        while len(class_num_str) < 3: class_num_str = '0' + class_num_str

        img = bytes_to_image(df.iloc[j][0]['bytes'])
        img_id = str(j)
        while len(img_id) < 6: img_id = '0' + img_id
        img_name = f'{class_num_str}_image_{img_id}.jpg'
        img_file_DST = f'{train_dir}/{img_name}'
        img.save(img_file_DST)

        meta_write_line = f'{img_name} {class_num}\n'
        train_meta_file.write(meta_write_line)
        train_meta_file.flush(); print(f'{meta_write_line}')

# Verify the start of index is 0
print('\nmin(train_labels): ', min(train_labels)) # 0
print('\nmax(train_labels): ', max(train_labels)) # 100

# Construct val set
val_labels = []
for i in range(3):
    df = pd.read_parquet(f'{data_root}/validation-0000{i}-of-00003.parquet')
    sh = df.shape
    for j in range(sh[0]):
        class_num = df.iloc[j][1]
        val_labels.append(class_num)
        class_num_str = str(class_num)
        while len(class_num_str) < 3: class_num_str = '0' + class_num_str

        img = bytes_to_image(df.iloc[j][0]['bytes'])
        img_id = str(j)
        while len(img_id) < 6: img_id = '0' + img_id
        img_name = f'{class_num_str}_image_{img_id}.jpg'
        img_file_DST = f'{val_dir}/{img_name}'
        img.save(img_file_DST)

        meta_write_line = f'{img_name} {class_num}\n'
        val_meta_file.write(meta_write_line)
        val_meta_file.flush(); print(f'{meta_write_line}')

# Verify the start of index is 0
print('\nmin(val_labels): ', min(val_labels))
print('\nmax(val_labels): ', max(val_labels))

# Clean other dirs
exec_cmd(f'rm -r {data_root}/*.parquet')

