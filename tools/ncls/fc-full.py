import os
import sys
from datetime import datetime
import argparse
import copy
import glob
from collections import defaultdict
import time
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
        parser.add_argument('-ns', '--n_subset', type=int, default=5)
        parser.add_argument('-mdr', '--meta_data_root', type=str, default='/datasets') # edit
        parser.add_argument('-gcp', '--gen_configs_path', type=str, default='./tools/ncls/gen_configs.yaml')
        parser.add_argument('-mdyfp', '--meta_data_yaml_file_path', type=str, default='./tools/ncls/datasets/ds.yaml') # edit
        parser.add_argument('-wd', '--work_dir', type=str, default='./work_dirs') # edit
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

        # to be removed later >>>
        '''
        if args.load_from_opt:
            self.data_root = f'{args.meta_data_root}/{args.dataset_name}'
        else:
            with open(args.meta_data_yaml_file_path, 'r') as f:
                data = yaml.safe_load(f)
                self.data_root = f'{data["meta_data_root"]}/{args.dataset_name}'
                print('\nself.data_root: ', self.data_root) 
        '''
        # to be removed later <<<

        # edit
        with open(self.gen_configs_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
            print('\nself.yaml_data: ', self.yaml_data)

        self.dataset_types = ['val'] # ['train', 'val']
        self.ncls_base_ls = [2, 3, 4, 5, 10]
        self.ncls_ratio = [0.1, 0.2, 0.4, 0.6, 0.8]
        self.ncls_ls = [] # to be updated

        # edit
        # to be updated later >>>
        '''
        self.model_to_ckp_path = defaultdict()
        self.checkpoints_path = './checkpoints'
        for ckp_path in glob.glob(f'{self.checkpoints_path}/*'):
            model = ckp_path.split('_')[0][len(self.checkpoints_path) + 1:]
            self.model_to_ckp_path[model] = ckp_path
        print(f'\nself.model_to_ckp_path: {self.model_to_ckp_path}')
        '''
        # to be updated later <<<

        self.eval_log_path = args.eval_log_path
        if not os.path.exists(self.eval_log_path): os.makedirs(self.eval_log_path)
        time_str = datetime.now().strftime('Y%YM%mD%d_H%HM%MS%S')
        # print('\ntime_str: ', time_str)
        self.eval_log_file_path = self.eval_log_path + f'/{time_str}.log'
        self.eval_log_file = open(self.eval_log_file_path, 'a')

def get_model_ls(path):
    model_ls = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            if len(line) > 0:
                model = line
                # print('\nmodel: ', model)
                model_ls.append(model)
    return model_ls

def gen_configs_and_eval():
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

            # model >>>
            for model in C.yaml_data['arch'][arch]['model']:
                print('\nmodel: ', model) # e.g. resnet18_8xb32_in1k
                config_root = C.yaml_data['arch'][arch]['path']
                print('\nconfig_root: ', config_root) # e.g. ./configs/resnet

                # Prepare ncls_ls
                if C.total_ncls_only:
                    C.ncls_ls = [total_ncls]
                    C.n_subset = 1
                else:
                    C.ncls_ls = copy.deepcopy(C.ncls_base_ls)
                    for r in C.ncls_ratio:
                        ncls = int(total_ncls * r)
                        C.ncls_ls.append(ncls)

                # ncls >>>
                for ncls in C.ncls_ls:
                    for seed in range(C.n_subset):
                        if C.total_ncls_only: seed = ''
                        print(f'\nncls: {ncls}, seed: {seed}')

                        # dataset >>>
                        # to be updated later >>>
                        # mainly changing ncls configs
                        '''
                        with open(config_ORI_path, 'r') as f:
                            dataset_content = f.read()
                        dataset_content_ = copy.deepcopy(dataset_content)
                        dataset_content_ = dataset_content_.replace('data_root', f'data_root=\'{C.data_root}\', #', 2) # train and val
                        dataset_content_ = dataset_content_.replace('train.txt', f'train_ncls_{ncls}_s_{seed}.txt')
                        dataset_content_ = dataset_content_.replace('val.txt', f'val_ncls_{ncls}_s_{seed}.txt')
                        '''
                        # to be updated later <<<
                        # dataset <<<

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

                        print('\narch_to_path: ', arch_to_path)
                        print('\nmodel_ORI_path: ', model_ORI_path)
                        print('\nspecs: ', specs)
                        print('\nmodel_DST_path: ', model_DST_path)

                        # if 'resnet' in model:
                        if True:
                            # Update config file >>>
                            '''
                            arch_to_path = C.yaml_data['arch'][arch]['path']
                            model_ORI_path = f'{arch_to_path}/{model}.py'
                            model_path = f'{arch_to_path}/{model}_ncls_{ncls}_s_{seed}'
                            model_DST_path = f'{model_path}.py'
                            print('\nmodel_ORI_path: ', model_ORI_path)
                            print('\nmodel_DST_path: ', model_DST_path)
                            ww
                            exec_cmd(f'scp {model_ORI_path} {model_DST_path}')
                            with open(model_DST_path, 'a') as f:
                                f.write('\n')
                                f.write(dataset_content_)
                                print(f'{model_DST_path} written!')
                            '''
                            # Update config file <<<          

                            # eval >>>
                            print(f'\nmodel: {model}')
                            # model_key = model.split('_')[0]
                            model_dataset_dir = f'{C.work_dir}/' + model_ORI_path[model_ORI_path.rindex('/') + 1:].replace('EDIT', dataset_name)[:-3]
                            print('\nmodel_dataset_dir: ', model_dataset_dir)
                            # e.g. ./configs/resnet/resnet50_8xb32_in1k_imagenet
                            
                            # if model_key in C.model_to_ckp_path:
                            path_ls = sorted(glob.glob(f'{model_dataset_dir}/best*.pth'), reverse=True)
                            print('\npath_ls: ', path_ls)

                            if len(path_ls) > 0: best_exists = True
                            else: best_exists = False

                            if not best_exists:
                                path_ls = sorted(glob.glob(f'{model_dataset_dir}/*.pth'), reverse=True)
                            ckp_path = path_ls[0]

                            # for ckp_path in glob.glob(f'{model_dataset_dir}/best*.pth'):
                            print('\nckp_path: ', ckp_path)
                            # www                            
                            # ckp_path = C.model_to_ckp_path[model_key]
                            
                            print(f'\nckp_path: {ckp_path}')
                            exec_cmd(f'CUDA_VISIBLE_DEVICES=2 python3 tools/test.py {model_DST_path} {ckp_path}')
                            # eval <<<

                            # Collect eval results >>>
                            eval_res_path = f'./work_dirs/{model}_ncls_{ncls}_s_{seed}_{dataset_name}'
                            logged = False
                            for wd_path in sorted(glob.glob(eval_res_path + '/*'), reverse=True):
                                if os.path.isdir(wd_path):
                                    eval_time = wd_path.split('/')[-1]
                                    eval_log_file_path = wd_path + f'/{eval_time}.log'
                                    with open(eval_log_file_path, 'r') as f:
                                        lines = f.readlines()
                                        for line in lines:
                                            if 'test' in line and 'accuracy/top1' in line and 'accuracy/top5' in line:
                                                str_ls = line.split(' ')
                                                '''
                                                for str_i, str_ in enumerate(str_ls):
                                                    print('\nstr_i: ', str_i, ', str_: ', str_)
                                                '''
                                                top1, top5 = str_ls[13], str_ls[16]

                                                # Write res to eval log file >>>
                                                line_write = f'{dataset_name}\t\t\t{model}\t\t\t{ncls}\t{seed}\t{top1}\t{top5}\n'
                                                C.eval_log_file.write(line_write)
                                                C.eval_log_file.flush()
                                                print(f'\n{line_write}Logged!')
                                                logged = True
                                                # Write res to eval log file <<<
                                                break
                                if logged: break
                            # Collect eval results <<<
                # ncls <<<
            # model <<<
        # arch <<<
    # dataset <<<

if __name__ == '__main__':
    gen_configs_and_eval()
