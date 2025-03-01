import os
import sys
from datetime import datetime
import argparse
import copy
import glob
from collections import defaultdict
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
        parser.add_argument('-mdyfp', '--meta_data_yaml_file_path', type=str, default='./tools/ncls/datasets/ds.yaml') # edit
        parser.add_argument('-wd', '--work_dir', type=str, default='./work_dirs') # edit
        parser.add_argument('-elp', '--eval_log_path', type=str, default='./work_dirs/eval')
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
        self.work_dir = args.work_dir

        # edit
        with open(self.gen_configs_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
            print('\nself.yaml_data: ', self.yaml_data)

        # edit
        self.model_to_ckp_path = defaultdict()
        self.checkpoints_path = './checkpoints'
        for ckp_path in glob.glob(f'{self.checkpoints_path}/*'):
            model = ckp_path.split('_')[0][len(self.checkpoints_path) + 1:]
            self.model_to_ckp_path[model] = ckp_path
        print(f'\nself.model_to_ckp_path: {self.model_to_ckp_path}')

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
    if C.dataset == 'imagenet':
        n_class_ls = [2, 3, 4, 5, 10, 100] #, 200, 400, 600, 800]
        n_subset = 5 # edit

    config_meta_path = './configs/_base_/datasets/imagenet_bs32'
    config_meta_path = './configs/_base_/datasets/imagenet_bs64_pil_resize.py'
    config_ORI_path = f'{config_meta_path}_ncls_1000.py'

    C.eval_log_file.write('Model\t\t\tncls\tseed\ttop1\ttop5\n')

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

        # model >>>
        for model in C.yaml_data['arch'][arch]['model']:
            print('\nmodel: ', model) # e.g. resnet18_8xb32_in1k

            for n_class in n_class_ls:
                for seed in range(n_subset):
                    print(f'\nn_class: {n_class}, seed: {seed}')

                    # dataset >>>
                    with open(config_ORI_path, 'r') as f:
                        dataset_content = f.read()
                    dataset_content_ = copy.deepcopy(dataset_content)
                    dataset_content_ = dataset_content_.replace('data_root', f'data_root=\'{C.data_root}\', #', 2) # train and val
                    dataset_content_ = dataset_content_.replace('train.txt', f'train_ncls_{n_class}_s_{seed}.txt')
                    dataset_content_ = dataset_content_.replace('val.txt', f'val_ncls_{n_class}_s_{seed}.txt')
                    # dataset <<<

                    # if 'resnet' in model:
                    if True:
                        # Update config file >>>
                        arch_to_path = C.yaml_data['arch'][arch]['path']
                        model_ORI_path = f'{arch_to_path}/{model}.py'
                        model_path = f'{arch_to_path}/{model}_ncls_{n_class}_s_{seed}'
                        model_DST_path = f'{model_path}.py'
                        exec_cmd(f'scp {model_ORI_path} {model_DST_path}')
                        with open(model_DST_path, 'a') as f:
                            f.write('\n')
                            f.write(dataset_content_)
                            print(f'{model_DST_path} written!')
                        # Update config file <<<          

                        # eval >>>
                        print(f'\nmodel: {model}')
                        model_key = model.split('_')[0]
                        if model_key in C.model_to_ckp_path:
                            ckp_path = C.model_to_ckp_path[model_key]
                            print(f'\nckp_path: {ckp_path}')
                            exec_cmd(f'python3 tools/test.py {model_DST_path} {ckp_path}')
                            # eval <<<

                            # Collect eval results >>>
                            eval_res_path = f'./work_dirs/{model}_ncls_{n_class}_s_{seed}'
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
                                                line_write = f'{model}\t\t\t{n_class}\t{seed}\t{top1}\t{top5}\n'
                                                C.eval_log_file.write(line_write)
                                                C.eval_log_file.flush()
                                                print(f'\n{line_write}Logged!')
                                                logged = True
                                                # Write res to eval log file <<<
                                                break
                                if logged: break
                            # Collect eval results <<<

if __name__ == '__main__':
    gen_configs_and_eval()
