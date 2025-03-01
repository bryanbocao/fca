import os
import yaml
'''
Find the weights in https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html
'''
def download_weights(path):
    checkpoints_path = './checkpoints'
    if not os.path.exists(checkpoints_path): os.makedirs(checkpoints_path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        for config, url in data.items():
            print('Downloading')
            print('\nconfig: ', config)
            print('\nurl: ', url)
            cmd = f'wget -P {checkpoints_path} {url}'
            os.system(cmd); print(cmd)

if __name__ == '__main__':
    config_to_url_path = './tools/ncls/config_to_url.yaml' # edit
    download_weights(config_to_url_path)
