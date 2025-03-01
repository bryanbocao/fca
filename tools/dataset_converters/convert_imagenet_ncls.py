'''
Usage:
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 2
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 3
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 4
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 5
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 10
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 100
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 200
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 400
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 600
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 800

python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt 
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 2
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 3
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 4
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 5
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 10
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 100
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 200
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 400
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 600
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 800
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
        parser.add_argument('-ncls', '--n_class', type=int, default=10)
        parser.add_argument('-ns', '--n_subset', type=int, default=5)
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
    for seed in range(C.args.n_subset):
        random.seed(seed)

        # sample a list of n_class
        cls_ls = random.sample(range(int(min(labels)), int(max(labels))), C.args.n_class)
        output_lines = []
        for k, l in zip(keys, labels):
            if int(l) in cls_ls:
                output_lines.append(f'{k} {l}\n')
        
        if 'train' in C.args.ann_path:
            write_path = C.args.meta_path + f'/train_ncls_{C.args.n_class}_s_{seed}.txt'
        elif 'val' in C.args.ann_path:
            write_path = C.args.meta_path + f'/val_ncls_{C.args.n_class}_s_{seed}.txt'
        with open(write_path, 'w+') as f:
            f.writelines(output_lines)
            print(f'{write_path} generated!')
        
if __name__ == '__main__':
    main()
