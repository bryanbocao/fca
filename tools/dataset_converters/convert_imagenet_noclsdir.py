'''
Usage:
python3 tools/dataset_converters/convert_imagenet_noclsdir.py -mp /datasets/imagenet/meta -ap train.txt
python3 tools/dataset_converters/convert_imagenet_noclsdir.py -mp /datasets/imagenet/meta -ap val.txt
'''

import random
import argparse

# -----------------------------------
#  Configurations for One Experiment
# -----------------------------------
class Config:
    def __init__(self):
        # --------------------------
        #  Paramaters of experiment
        # --------------------------
        parser = argparse.ArgumentParser()
        parser.add_argument('-mp', '--meta_path', type=str, default=None, help='Path to the root of meta')
        parser.add_argument('-ap', '--ann_path', type=str, default=None, help='Path to the annotation file.')
        self.args = parser.parse_args()

def main():
    C = Config()

    # create dict with full imagenet annotation file
    read_path = C.args.meta_path + '/' + C.args.ann_path
    with open(read_path, 'r') as f:
        lines = f.readlines()
    keys = [line.split(' ')[0] for line in lines]
    labels = [line.strip().split()[1] for line in lines]
    mapping = {}
    for k, l in zip(keys, labels):
        if k not in mapping:
            mapping[k] = l
        else:
            assert mapping[k] == l
    
    # convert
    output_lines = []
    for k, l in zip(keys, labels):
        if C.args.ann_path == 'train.txt':
            k = k.split('/')[1]
        output_lines.append(f'{k} {l}\n')
    
    if C.args.ann_path == 'train.txt':
        write_path = C.args.meta_path + f'/train_noclsdir.txt'
    elif C.args.ann_path == 'val.txt':
        write_path = C.args.meta_path + f'/val_noclsdir.txt'
    with open(write_path, 'w+') as f:
        f.writelines(output_lines)
        print(f'{write_path} generated!')
        
if __name__ == '__main__':
    main()
