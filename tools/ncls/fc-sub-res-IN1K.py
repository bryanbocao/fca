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
        parser.add_argument('-tnclso', '--total_ncls_only', action='store_true', default=False) # edit
        parser.add_argument('-b', '--best', action='store_true', default=False, help='Evaluate the best checkpoint. Otherwise the last checkpoint') # edit
        parser.add_argument('-ns', '--n_subset', type=int, default=5) # edit
        parser.add_argument('-mdr', '--meta_data_root', type=str, default='/datasets') # edit
        parser.add_argument('-gcp', '--gen_configs_path', type=str, default='./tools/ncls/gen_configs.yaml')
        parser.add_argument('-mdyfp', '--meta_data_yaml_file_path', type=str, default='./tools/ncls/datasets/ds.yaml') # edit
        parser.add_argument('-wd', '--work_dir', type=str, default='./work_dirs') # edit # where trained models (checkpoints) are saved
        parser.add_argument('-elp', '--eval_log_path', type=str, default='./work_dirs/eval')
        args = parser.parse_args()

        self.load_from_opt = args.load_from_opt
        self.dataset_name = args.dataset_name
        self.total_ncls_only = args.total_ncls_only
        self.best = args.best
        self.n_subset = args.n_subset
        self.meta_data_root = args.meta_data_root
        self.meta_data_yaml_file_path = args.meta_data_yaml_file_path
        self.gen_configs_path = args.gen_configs_path
        self.work_dir = args.work_dir

        # edit
        with open(self.gen_configs_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
            print('\nself.yaml_data: ', self.yaml_data)

        self.dataset_types = ['val'] # ['train', 'val']
        self.ncls_base_ls = [2, 3, 4, 5, 10]
        self.ncls_ratio = [0.1, 0.2, 0.4, 0.6, 0.8]
        self.ncls_ls = [] # to be updated

        self.eval_log_path = args.eval_log_path
        if not os.path.exists(self.eval_log_path): os.makedirs(self.eval_log_path)
        time_str = datetime.now().strftime('Y%YM%mD%d_H%HM%MS%S')
        # print('\ntime_str: ', time_str)
        self.eval_log_file_path = self.eval_log_path + f'/{time_str}.log'
        self.eval_log_file = open(self.eval_log_file_path, 'a')


def gen_res():
    C = Config()
    time_stamp = datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d_%H_%M_%S")
    C.eval_log_file.write('Dataset_Name\t\t\tModel\t\t\tncls\tseed\ttop1\ttop5\n')

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

        # arch >>>
        print('\nC.yaml_data[\'arch\'].keys(): ', C.yaml_data['arch'].keys())
        # e.g. dict_keys(['resnet', 'vision_transformer'])
        for arch in C.yaml_data['arch'].keys():
            print('\narch: ', arch) # e.g. resnet
           
            for model in C.yaml_data['arch'][arch]['model']:
                print('\nmodel: ', model) # e.g. resnet18_8xb32_in1k

                # Prepare ncls_ls
                if C.total_ncls_only:
                    C.ncls_ls = [total_ncls]
                    C.n_subset = 1
                else:
                    C.ncls_ls = copy.deepcopy(C.ncls_base_ls)
                    for r in C.ncls_ratio:
                        ncls = int(total_ncls * r)
                        C.ncls_ls.append(ncls)

                for ncls in C.ncls_ls:
                    for seed in range(C.n_subset):
                        if C.total_ncls_only: seed = 'None'
                        print(f'\nncls: {ncls}, seed: {seed}')

                        ###
                        # Collect eval results >>>
                        if C.total_ncls_only: eval_res_path = f'{C.work_dir}/{model}_{dataset_name}'
                        else: eval_res_path = f'{C.work_dir}/{model}_ncls_{ncls}_s_{seed}_{dataset_name}'
                        print(f'\neval_res_path: {eval_res_path}')
                        
                        logged = False
                        if C.best: res_next_line = False
                        else: res_next_line = True
                        trained_dirs = sorted(glob.glob(eval_res_path + '/*'), reverse=True)
                        print(f'\n trained_dirs: {trained_dirs}')
                        if len(trained_dirs) > 0:
                            for wd_path in trained_dirs:
                                if os.path.isdir(wd_path) and 'last_checkpoint' not in wd_path:
                                    print('\nwd_path: ', wd_path)
                                    eval_time = wd_path.split('/')[-1]
                                    eval_log_file_path = wd_path + f'/{eval_time}.log'
                                    print(f'eval_log_file_path: {eval_log_file_path}')
                                    with open(eval_log_file_path, 'r') as f:
                                        lines = f.readlines()
                                        line_i = len(lines) - 1
                                        # for line in lines:
                                        while line_i >= 0:
                                            line = lines[line_i]
                                            if C.best and 'best' in line:
                                                res_next_line = True
                                            if res_next_line and 'Epoch(val)' in line and 'accuracy/top1' in line and 'accuracy/top5' in line:
                                                str_ls = line.split(' ')
                                                '''
                                                for str_i, str_ in enumerate(str_ls):
                                                    print('\nstr_i: ', str_i, ', str_: ', str_)
                                                '''
                                                top1, top5 = str_ls[13], str_ls[16]

                                                # Write res to eval log file >>>
                                                line_write = f'{dataset_name}\t{model}\t{ncls}\t{seed}\t{top1}\t{top5}\n'
                                                C.eval_log_file.write(line_write)
                                                C.eval_log_file.flush()
                                                print(f'\n{line_write}Logged!')
                                                logged = True
                                                # Write res to eval log file <<<
                                                break
                                            line_i -= 1
                                if logged: break
                        # Collect eval results <<<
        # arch <<<
    # dataset <<<

if __name__ == '__main__':
    gen_res()
