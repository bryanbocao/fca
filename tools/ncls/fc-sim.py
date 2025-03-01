import os
import sys
sys.path.append('./tools')
sys.path.append('./configs/_base_')
from datetime import datetime
from utils import exec_cmd
from sim import Similarity
import random
import numpy as np
import glob
import yaml
import copy
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel

from PIL import Image 
from mmpretrain import get_model
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
        parser.add_argument('-mdsyp', '--meta_data_yaml_file_path', type=str, default='./tools/ncls/datasets/ds.yaml') # edit
        parser.add_argument('-nclsyp', '--ncls_yaml_path', type=str, default='./tools/ncls/ncls_datasets_models_EDIT.yaml') # edit
        parser.add_argument('-dsn', '--dataset_name', type=str, default=None) # edit
        parser.add_argument('-ns', '--n_subset', type=int, default=5)
        parser.add_argument('-sb', '--sim_base', type=str, default=None, help='dinov2 | clip') # edit
        parser.add_argument('-elp', '--sim_log_path', type=str, default='./work_dirs/sim')
        args = parser.parse_args()

        self.meta_data_yaml_file_path = args.meta_data_yaml_file_path
        self.ncls_yaml_path = args.ncls_yaml_path
        self.dataset_name = args.dataset_name
        self.meta_path = '' # to be updated
        self.anno_path = '' # to be updated
        self.read_path = '' # to be updated
        self.n_subset = args.n_subset
        self.sim_base = args.sim_base
        self.sim_base_ls = ['dinov2', 'clip'] # edit

        with open(self.meta_data_yaml_file_path, 'r') as f:
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
        self.dataset_types = ['val'] # ['train', 'val'] only 'val' for similarity
        self.ncls_base_ls = [2, 3, 4, 5, 10]
        self.ncls_ratio = [0.1, 0.2, 0.4, 0.6, 0.8]
        self.ncls_ls = [] # to be updated

        self.sim = None

        self.sim_log_path = args.sim_log_path
        if not os.path.exists(self.sim_log_path): os.makedirs(self.sim_log_path)
        time_str = datetime.now().strftime('Y%YM%mD%d_H%HM%MS%S')
        # print('\ntime_str: ', time_str)
        self.sim_log_folder = self.sim_log_path + f'/{time_str}' # Will be updated
        if not os.path.exists(self.sim_log_folder): os.makedirs(self.sim_log_folder)
        self.sim_log_file_path = None # Will be updated
        self.sim_log_file = None # Will be updated; open(self.sim_log_file_path, 'a')

def main():
    C = Config()
    print('\nC.sim_base: ', C.sim_base)
    print('\nC.sim_base_ls: ', C.sim_base_ls)
    if C.sim_base is not None:
        C.sim_base_ls = [C.sim_base]

    for sim_base in C.sim_base_ls:
        print('\nsim_base: ', sim_base)

        dataset_from_opt = False
        for ds_i, dataset in enumerate(C.datasets):
            if C.dataset_name is not None:
                dataset_name = C.dataset_name
                while dataset_name not in C.datasets[ds_i]: ds_i += 1
                dataset = C.datasets[ds_i]
                dataset_from_opt = True

            print('\ndataset: ', dataset)
            dataset_name = list(dataset.keys())[0]
            print('\ndataset_name: ', dataset_name)

            total_ncls = dataset[dataset_name]['ncls']
            print('\ntotal_ncls: ', total_ncls)

            # create dict with full imagenet annotation file
            C.meta_path = f'{C.meta_data_root}/{dataset_name}/meta'; print('\nC.meta_path: ', C.meta_path)
            C.sim_log_file_path = C.sim_log_folder + f'/{dataset_name}_{sim_base}.log'
            C.sim_log_file = open(C.sim_log_file_path, 'a')
            C.sim_log_file.write('ncls\tseed\ts_alpha\ts_beta\ts_SS\n')
            C.sim = Similarity(sim_base) # for each dataset

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

                # Iterate over all ncls subsets
                for ncls in C.ncls_ls:
                    # convert
                    for seed in range(C.n_subset):
                        random.seed(seed)

                        # Sample a list of n_class
                        print('\nrange(int(min(labels)), int(max(labels))): ', range(int(min(labels)), int(max(labels))))
                        print('\nncls: ', ncls)
                        print('\ndataset: ', dataset)
                        cls_ls = random.sample(range(int(min(labels)), int(max(labels))), ncls)

                        # Read images and cache all
                        C.read_path = C.meta_path + f'/{ds_type}_ncls_{ncls}_s_{seed}.txt'
                        with open(C.read_path, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                print('\nline: ', line) # e.g. 245_image_0010.jpg 245
                                img_id, cls = line.split(' '); cls = cls[:-1]
                                print('\nimg_id: ', img_id)
                                print('\ncls: ', cls)

                                # Cache img feats
                                if img_id not in C.sim.img_id_to_feats_dict:
                                    img_path = f'{C.meta_data_root}/{dataset_name}/{ds_type}/{img_id}'
                                    img = Image.open(img_path) 
                                    feats = C.sim.extract_feats(img)
                                    C.sim.img_id_to_feats_dict[img_id] = feats

                                # Save cls_id to img_id ls
                                if cls not in C.sim.cls_id_to_img_id_ls_dict: C.sim.cls_id_to_img_id_ls_dict[cls] = []
                                else: C.sim.cls_id_to_img_id_ls_dict[cls].append(img_id)

                        s_alpha = C.sim.sim_alpha(cls_ls) # Intra-Class Similarity
                        s_beta = C.sim.sim_beta(cls_ls) # Inter-Class Similarity
                        s_SS = C.sim.sim_SS(cls_ls) # SimSS: Similarity-Based Silhouette Score
                        print(f'\ns_alpha: {s_alpha}, s_beta: {s_beta}, s_SS: {s_SS}')

                        # Write res to eval log file
                        line_write = f'{ncls}\t{seed}\t{s_alpha}\t{s_beta}\t{s_SS}\n'
                        C.sim_log_file.write(line_write)
                        C.sim_log_file.flush()
                        print(f'\n{line_write}Logged!')
        if dataset_from_opt: break

if __name__ == '__main__':
    main()

