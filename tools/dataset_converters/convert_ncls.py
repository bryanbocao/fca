import os
import glob
import sys
sys.path.append('./tools')
from utils import exec_cmd
import random
import yaml
import copy
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
        parser.add_argument('-dsyp', '--ds_yaml_path', type=str, default='./tools/ncls/datasets/ds.yaml') # edit
        parser.add_argument('-nclsyp', '--ncls_yaml_path', type=str, default='./tools/ncls/ncls_datasets_models_EDIT.yaml') # edit
        parser.add_argument('-ns', '--n_subset', type=int, default=5)
        args = parser.parse_args()

        self.ds_yaml_path = args.ds_yaml_path
        self.ncls_yaml_path = args.ncls_yaml_path
        self.meta_path = '' # to be updated
        self.anno_path = '' # to be updated
        self.write_path = '' # to be updated
        self.n_subset = args.n_subset


        with open(self.ds_yaml_path, 'r') as f:
            yaml_file_data = yaml.safe_load(f)
            self.meta_data_root = yaml_file_data['meta_data_root']
            print('\nself.meta_data_root: ', self.meta_data_root)

        with open(self.ncls_yaml_path, 'r') as f:
            yaml_file_data = yaml.safe_load(f)
            self.datasets = yaml_file_data['datasets']
            print('\nself.datasets: ', self.datasets)

        '''
        self.meta_data_root:  /datasets
        self.datasets:  [{'caltech101': {'ncls': 101}}, {'caltech256': {'ncls': 257}}, {'cifar100': {'ncls': 100}}, {'cub200': {'ncls': 200}}, {'food101': {'ncls': 101}}, {'gtsrb43': {'ncls': 43}}, {'indoor67': {'ncls': 67}}, {'quickdraw345': {'ncls': 345}}, {'textures47': {'ncls': 47}}]
        '''
        self.dataset_types = ['train', 'val']
        self.ncls_base_ls = [2, 3, 4, 5, 10]
        self.ncls_ratio = [0.1, 0.2, 0.4, 0.6, 0.8]
        self.ncls_ls = [] # to be updated

def main():
    C = Config()
    
    for ds_i, dataset in enumerate(C.datasets):
        print('\ndataset: ', dataset)
        dataset_name = list(dataset.keys())[0]
        print('\ndataset_name: ', dataset_name)

        total_ncls = dataset[dataset_name]['ncls']
        print('\ntotal_ncls: ', total_ncls)

        # create dict with full imagenet annotation file
        C.meta_path = f'{C.meta_data_root}/{dataset_name}/meta'; print('\nC.meta_path: ', C.meta_path)
        for ds_type in C.dataset_types:
            C.anno_path = f'{C.meta_path}/{ds_type}_noclsdir.txt'
            exec_cmd(f'scp -r {C.meta_path}/{ds_type}.txt {C.anno_path}')
            print('\nC.meta_path: ', C.meta_path)

            with open(C.anno_path, 'r') as f:
                lines = f.readlines()
            keys = [line.split(' ')[0] for line in lines]
            labels = [int(line.strip().split()[1]) for line in lines]
            
            # print('\nlabels: ', labels)
            # print('\nint(max(labels)): ', int(max(labels)))
            mapping = {}
            for k, l in zip(keys, labels):
                if k not in mapping:
                    mapping[k] = l
                else:
                    assert mapping[k] == l      

            # Prepare ncls_ls
            C.ncls_ls = copy.deepcopy(C.ncls_base_ls)
            for r in C.ncls_ratio:
                ncls = int(total_ncls * r)
                C.ncls_ls.append(ncls)

            for ncls in C.ncls_ls:
                # convert
                for seed in range(C.n_subset):
                    random.seed(seed)

                    # sample a list of n_class
                    print('\nrange(int(min(labels)), int(max(labels))): ', range(int(min(labels)), int(max(labels))))
                    print('\nncls: ', ncls)
                    print('\ndataset: ', dataset)
                    cls_ls = random.sample(range(int(min(labels)), int(max(labels))), ncls)
                    output_lines = []
                    for k, l in zip(keys, labels):
                        if int(l) in cls_ls:
                            output_lines.append(f'{k} {l}\n')
                        
                    C.write_path = C.meta_path + f'/{ds_type}_ncls_{ncls}_s_{seed}.txt'
                    with open(C.write_path, 'w+') as f:
                        f.writelines(output_lines)
                        print(f'{C.write_path} generated!')
        
if __name__ == '__main__':
    main()
