import os
import argparse
import copy
import glob
from collections import defaultdict
import time
from datetime import datetime
import sys
sys.path.append('./tools')
from utils import exec_cmd
import yaml

# -----------------------------------
#  Configurations for One Experiment
# -----------------------------------
class Config:
    def __init__(self):
        # --------------------------
        #  Paramaters of experiment
        # --------------------------
        parser = argparse.ArgumentParser()
        parser.add_argument('-o', '--load_from_opt', action='store_true', default=False)
        parser.add_argument('-ds', '--dataset', type=str, default='imagenet', help='imagenet | cifar10 | cifar100') # edit
        parser.add_argument('-mdr', '--meta_data_root', type=str, default='/datasets') # edit
        parser.add_argument('-gcp', '--gen_configs_path', type=str, default='./tools/ncls/gen_configs.yaml')
        parser.add_argument('-btp', '--batch_train_path', type=str, default='./tools/ncls/batch_train.sh')
        parser.add_argument('-ns', '--n_subset', type=int, default=5) # edit
        parser.add_argument('-mdyfp', '--meta_data_yaml_file_path', type=str, default='./tools/ncls/datasets/ds.yaml') # edit
        args = parser.parse_args()

        if args.load_from_opt:
            self.data_root = f'{args.meta_data_root}/{args.dataset}'
        else:
            with open(args.meta_data_yaml_file_path, 'r') as f:
                data = yaml.safe_load(f)
                self.data_root = f'{data["meta_data_root"]}/{args.dataset}'
                print('\nself.data_root: ', self.data_root)

        self.dataset = args.dataset
        self.gen_configs_path = args.gen_configs_path
        self.batch_train_path = args.batch_train_path
        self.n_subset = args.n_subset

        # edit
        with open(self.gen_configs_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
            print('\nself.yaml_data: ', self.yaml_data)


def gen_configs():
    C = Config()
    time_stamp = datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d_%H_%M_%S")
    write_f = open(C.batch_train_path, 'w')
    write_f_content = ''

    training_logs_dir = './training_logs'
    if not os.path.exists(training_logs_dir):
        os.makedirs(training_logs_dir); print(f'{training_logs_dir} created!')

    if C.dataset == 'imagenet':
        n_class_ls = [2, 3, 4, 5, 10, 100, 200, 400, 600, 800]

    config_meta_path = './configs/_base_/datasets/imagenet_bs32' # debug
    config_meta_path = './configs/_base_/datasets/imagenet_bs64_pil_resize.py' # debug
    config_ORI_path = f'{config_meta_path}_ncls_1000.py' # debug
    # model_ls = get_model_ls(C.gen_configs_ls_path)

    # runtime
    # runtime_path = 'configs/_base_/ncls_runtime.py' # edit
    runtime_path = 'configs/_base_/default_runtime_save_best.py' # edit

    cuda_id = 0
    # arch >>>
    print('\nC.yaml_data[\'arch\'].keys(): ', C.yaml_data['arch'].keys())
    # e.g. dict_keys(['resnet', 'vision_transformer'])
    for arch in C.yaml_data['arch'].keys():
        print('\narch: ', arch) # e.g. resnet
        if 'resnet' in arch:
            config_meta_path = './configs/_base_/datasets/imagenet_bs32'
        elif 'vision_transformer' in arch:
            config_meta_path = './configs/_base_/datasets/imagenet_bs64_pil_resize'
        config_ORI_path = f'{config_meta_path}_ncls_1000.py'

        for model in C.yaml_data['arch'][arch]['model']:
            print('\nmodel: ', model) # e.g. resnet18_8xb32_in1k
            for n_class in n_class_ls:
                for seed in range(C.n_subset):
                    print(f'\nn_class: {n_class}, seed: {seed}')

                    # dataset >>>
                    with open(config_ORI_path, 'r') as f:
                        dataset_content = f.read()
                    dataset_content_ = copy.deepcopy(dataset_content)
                    dataset_content_ = dataset_content_.replace('data_root', f'data_root=\'{C.data_root}\', #', 2) # train and val
                    dataset_content_ = dataset_content_.replace('train.txt', f'train_ncls_{n_class}_s_{seed}.txt')
                    dataset_content_ = dataset_content_.replace('val.txt', f'val_ncls_{n_class}_s_{seed}.txt')
                    # dataset <<<

                    # runtime >>>
                    with open(runtime_path, 'r') as f:
                        runtime_content = f.read()
                    # runtime <<<

                    # Update config file >>>
                    arch_to_path = C.yaml_data['arch'][arch]['path']
                    if 'vit' in model:
                        model_ORI_path = f'{arch_to_path}/{model}_EDIT.py'
                    else:
                        model_ORI_path = f'{arch_to_path}/{model}.py'
                    model_path = f'{arch_to_path}/{model}_ncls_{n_class}_s_{seed}'
                    model_DST_path = f'{model_path}.py'
                    exec_cmd(f'scp {model_ORI_path} {model_DST_path}')
                    with open(model_DST_path, 'a') as f:
                        f.write('\n')
                        f.write(dataset_content_)
                        f.write(runtime_content); print(f'{model_DST_path} written!')
                    # Update config file <<<

                    # train file >>>
                    cmd = f'CUDA_VISIBLE_DEVICES={cuda_id} nohup python3 tools/train.py {model_DST_path} --amp > {training_logs_dir}/{model}_ncls_{n_class}_s_{seed}_{time_stamp}.log & \n'
                    write_f_content += cmd
                    # train file <<<
                    cuda_id = (cuda_id + 1) % 8 # 8 GPUs
    # arch <<<
    write_f.write(write_f_content); print(f'{C.batch_train_path} written!')

if __name__ == '__main__':
    gen_configs()
