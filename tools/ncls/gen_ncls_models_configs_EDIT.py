import os
import glob
import sys
sys.path.append('./tools')
from utils import exec_cmd
import yaml

with open('./tools/ncls/datasets/ds.yaml', 'r') as f:
    ds_yaml_file_data = yaml.safe_load(f)
    meta_data_root = ds_yaml_file_data['meta_data_root']
    print('\nmeta_data_root: ', meta_data_root)
    # e.g. meta_data_root:  /datasets

with open('./tools/ncls/ncls_datasets_models_EDIT.yaml', 'r') as f:
    data = yaml.safe_load(f)
    print('\ndata: ', data)

for dataset in data['datasets']:
    print('\ndataset: ', dataset)
    dataset_name = list(dataset.keys())[0]
    print('\ndataset_name: ', dataset_name)
    data_root = f'{meta_data_root}/{dataset_name}'
    print('\ndata_root: ', data_root)
    # e.g. data_root:  /datasets/imagenet

    for model, ORI_config_file_path in data['models'].items():
        print('\n===========================')
        print('\nmodel: ', model)
        print('\nORI_config_file_path: ', ORI_config_file_path)
        # e.g. ORI_config_file_path:  ./configs/resnet/resnet50_8xb32_in1k_EDIT.py
        
        path_ls = ORI_config_file_path.split('/')
        config_root = f'./{path_ls[1]}/{path_ls[2]}'
        print('\nconfig_root: ', config_root)
        # e.g. config_root:  ./configs/resnet

        def edit_file(config_file_path):
            # DST_config_file_path
            DST_config_file_path = config_file_path.replace('EDIT', dataset_name)
            print('\nDST_config_file_path: ', DST_config_file_path)
            # e.g. DST_config_file_path:  ./configs/resnet/resnet50_8xb32_in1k_imagenet.py
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
                                path = config_root + '/' + line[line.index('\'')+1:line.index('.py')+len('.py')]
                                edit_file(path)
                            
                            line = line.replace('EDIT', dataset_name)
                        if 'edit' in line:
                            if 'num_classes=' in line:
                                ncls = dataset[dataset_name]['ncls']
                                line = line.replace('num_classes=', f'num_classes={ncls}')
                            if 'data_root' in line:
                                line = line.replace("data_root=''", f"data_root='{meta_data_root}/{dataset_name}'")
                        DST_config_file.write(line); DST_config_file.flush(); print(line)
        
        # edit file
        edit_file(ORI_config_file_path)

