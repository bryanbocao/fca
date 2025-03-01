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
        parser.add_argument('-dsn', '--dataset_name', type=str, default='imagenet', help='imagenet | cifar10 | cifar100') # edit
        parser.add_argument('-ns', '--n_subset', type=int, default=5) # edit
        parser.add_argument('-mdr', '--meta_data_root', type=str, default='/datasets') # edit
        parser.add_argument('-gcp', '--gen_configs_path', type=str, default='./tools/ncls/gen_configs.yaml')
        parser.add_argument('-mdyfp', '--meta_data_yaml_file_path', type=str, default='./tools/ncls/datasets/ds.yaml') # edit
        parser.add_argument('-btp', '--batch_train_path', type=str, default='./tools/ncls/batch_train.sh')
        args = parser.parse_args()

        self.load_from_opt = args.load_from_opt
        self.dataset_name = args.dataset_name
        self.n_subset = args.n_subset
        self.meta_data_root = args.meta_data_root
        self.meta_data_yaml_file_path = args.meta_data_yaml_file_path
        self.gen_configs_path = args.gen_configs_path
        self.batch_train_path = args.batch_train_path
        self.n_subset = args.n_subset

        # edit
        with open(self.gen_configs_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
            print('\nself.yaml_data: ', self.yaml_data)

        self.dataset_types = ['train'] # ['train', 'val']
        self.ncls_base_ls = [2, 3, 4, 5, 10]
        self.ncls_ratio = [0.1, 0.2, 0.4, 0.6, 0.8]
        self.ncls_ls = [] # to be updated

        self.time_stamp = datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d_%H_%M_%S")
        self.batch_train_path = self.batch_train_path[:self.batch_train_path.index('.sh')] + '_' + self.time_stamp + '.sh'
        self.write_f = open(self.batch_train_path, 'w')
        self.write_f_content = ''

def gen_configs():
    C = Config()

    training_logs_dir = './training_logs'
    if not os.path.exists(training_logs_dir):
        os.makedirs(training_logs_dir); print(f'{training_logs_dir} created!')

    # dataset >>>
    for dataset in C.yaml_data['datasets']:
        print('\ndataset: ', dataset)
        dataset_name = list(dataset.keys())[0]
        print('\ndataset_name: ', dataset_name)

        total_ncls = dataset[dataset_name]['ncls']
        print('\ntotal_ncls: ', total_ncls)
        print('\ndataset_name: ', dataset_name)
        if C.load_from_opt:
            C.data_root = f'{C.meta_data_root}/{C.dataset_name}'
        else:
            with open(C.meta_data_yaml_file_path, 'r') as f:
                data = yaml.safe_load(f)
                C.meta_data_root = data['meta_data_root']
                C.data_root = f'{C.meta_data_root}/{C.dataset_name}'
        print('\nC.data_root: ', C.data_root)
        # e.g. C.data_root:  /datasets/imagenet

        cuda_id = 0
        # arch >>>
        print('\nC.yaml_data[\'arch\'].keys(): ', C.yaml_data['arch'].keys())
        # e.g. dict_keys(['resnet', 'vision_transformer'])
        for arch in C.yaml_data['arch'].keys():
            print('\narch: ', arch) # e.g. resnet

            for model in C.yaml_data['arch'][arch]['model']:
                print('\nmodel: ', model) # e.g. resnet18_8xb32_in1k
                config_root = C.yaml_data['arch'][arch]['path']
                print('\nconfig_root: ', config_root) # e.g. ./configs/resnet  

                # Prepare ncls_ls
                C.ncls_ls = copy.deepcopy(C.ncls_base_ls)
                for r in C.ncls_ratio:
                    ncls = int(total_ncls * r)
                    C.ncls_ls.append(ncls)

                for ncls in C.ncls_ls:
                    for seed in range(C.n_subset):
                        print(f'\nncls: {ncls}, seed: {seed}')

                        # Update config file >>>
                        def edit_file(config_file_path, specs):
                            DST_config_file_path = None
                            if os.path.exists(config_file_path):
                                # DST_config_file_path
                                DST_config_file_path = config_file_path.replace('EDIT', specs)
                                print('\nDST_config_file_path: ', DST_config_file_path)
                                # e.g. DST_config_file_path:  ./configs/mobilevit/mobilevit-small_8xb128_in1k_ncls_37_s_1_textures47.py
                                exec_cmd(f'rm {DST_config_file_path}')
                                exec_cmd(f'touch {DST_config_file_path}')

                                DST_config_file = open(DST_config_file_path, 'a')
                                with open(config_file_path, 'r') as f:
                                    lines = f.readlines()
                                    for line in lines:
                                        if line != '\n':
                                            if 'EDIT' in line:
                                                if '.py' in line:
                                                    line_py = line.replace('..', './configs')
                                                    print('\nspecs: ', specs) # e.g. ncls_2_s_0_imagenet
                                                    print('\nline_py: ', line_py)
                                                    path = config_root + '/' + line[line.index('\'')+1:line.index('.py')+len('.py')]
                                                    print('\npath: ', path) # e.g. ./configs/resnet/../_base_/models/resnet50_EDIT.py
                                                    edit_file(path, specs)

                                                line = line.replace('EDIT', dataset_name)
                                            if 'edit' in line:
                                                if 'num_classes=' in line:
                                                    # Calculate the number of left brackets to fill right brackets after editing
                                                    n_bracket = 0
                                                    ncls_idx = line.index('num_classes')
                                                    for c in line[:ncls_idx]:
                                                        if c == '(': n_bracket += 1
                                                        elif c == ')': n_bracket -= 1

                                                    # Check if comma should be added later
                                                    comma = False
                                                    for c in line[ncls_idx:]:
                                                        if c == ',': comma = True

                                                    line = line[:ncls_idx + len('num_classes')]
                                                    line = line + f'={total_ncls}'
                                                    for b in range(n_bracket):
                                                        line = line + ')'
                                                    if comma: line = line + ', # edit \n'
                                                    else: line = line + ' # edit \n'
                                                    '''
                                                    if '(' in line:
                                                        line = line.replace('num_classes=', f'num_classes={total_ncls}), #')
                                                    else:
                                                        line = line.replace('num_classes=', f'num_classes={total_ncls}, #')
                                                    '''
                                                if 'data_root' in line:
                                                    line = line.replace("data_root=''", f"data_root='{C.meta_data_root}/{dataset_name}'")
                                                if 'ann_file' in line:
                                                    if 'train.txt' in line:
                                                        line = line.replace('train.txt', f'train_ncls_{ncls}_s_{seed}.txt')
                                                    if 'val.txt' in line:
                                                        line = line.replace('val.txt', f'val_ncls_{ncls}_s_{seed}.txt')
                                            DST_config_file.write(line); DST_config_file.flush(); print(line)
                            return DST_config_file_path
                        arch_to_path = C.yaml_data['arch'][arch]['path']
                        model_ORI_path = f'{arch_to_path}/{model}_EDIT.py'

                        specs = f'{dataset_name}'
                        model_DST_path = edit_file(model_ORI_path, specs)

                        specs = f'ncls_{ncls}_s_{seed}_{dataset_name}'
                        model_DST_path = edit_file(model_ORI_path, specs)
                        # Update config file <<<

                        # train file >>>
                        if model_DST_path:
                            cmd = f'CUDA_VISIBLE_DEVICES={cuda_id} nohup python3 tools/train.py {model_DST_path} --amp > {training_logs_dir}/{model}_ncls_{ncls}_s_{seed}_{dataset_name}_{C.time_stamp}.log & \n'
                            C.write_f_content += cmd
                        # train file <<<
                        cuda_id = (cuda_id + 1) % 8 # 8 GPUs
        # arch <<<
    # dataset <<<
    C.write_f.write(C.write_f_content); print(f'{C.batch_train_path} written!')

if __name__ == '__main__':
    gen_configs()
