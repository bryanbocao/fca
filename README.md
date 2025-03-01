# <div align="center"> Few-Class Arena </div>

#### News
- 2025.01.22 Our Few-Class Arena (FCA) paper ([arXiv 2411.01099](https://arxiv.org/pdf/2411.01099) | [OpenReview](https://openreview.net/pdf?id=2ET561DyPe)) has been accepted to **ICLR 2025** 😄!

Authors: Bryan Bo Cao, Lawrence O’Gorman, Michael Coss, Shubham Jain

A platform for conducting research in the few-class regime.

[Introduction](#introduction) |
[Major Features](#major-features) |
[Installation](#installation) |
[Versions](#versions) |
[Repository Structure](#repository-structure) |
[User Guidelines](#user-guidelines)


## Introduction
<div aligh="center">
<img align="center" width="640" alt="overview" src="https://github.com/user-attachments/assets/65f64a13-8efd-4b3d-b0fa-221dbbe7e5ca">
</div>

Few-Class-Arena (FCA) is an open platform written in PyTorch developed on top of the [OpenMMLab](https://openmmlab.com/) project. It provides an open source toolbox for conducting research in the few-class regime (classification and detection systems whose dataset consists of few classes, typically <10). FCA encapsulates the underlying tedious coding and configurations for each experiment, and it provides a convenient interface for users to conduct large-scale experiments in batch. It saves a large amount of time for researchers by omitting the steps of manually conducting experiments and gathering results from each individual experiment independently. Users can enjoy these features by specifying the configurations for different tasks including training and evaluation.

![fca_fig1](https://github.com/fewclassarena/fca/assets/165857143/c9f7ed91-c04d-49ce-a814-8d2eda18c353)
Top-1 accuracies of various scales of ResNet, whose model sizes are shown in the legend, and whose plots vary from dark to light by decreasing size. Plots range along number of classes N<sub>CL</sub> from the full ImageNet size (1000) down to the _Few-Class Regime_. Each model is tested on 5 subsets whose N<sub>CL</sub> classes are randomly sampled from the original 1000 classes. (a) Plots for sub-models trained on subsets of classes (blue) and full models trained on all 1000 classes (red). (b) Zoomed window shows the standard deviation of subset’s accuracies is much smaller than for the full model. (c.1) Full model accuracies drop when N<sub>CL</sub> decreases. (c.2) Full model accuracies increase as model scales up in the _Few-Class Regime_. (d.1) Sub-model accuracies grow as N<sub>CL</sub> decreases. (d.2) Sub-model accuracies do not increase when model scales up in the _Few-Class Regime_.

![fca_fig5](https://github.com/fewclassarena/fca/assets/165857143/ef315361-21a2-464e-9f1d-fd77d42164cb)
Pearson correlation coefficient (r) between DCN and SimSS when N<sub>Cl</sub> ∈ {2, 3, 4, 5, 10, 100}. DCN-Sub (blue squares) is more highly correlated than DCN-Full (red diamonds) with SimSS using both similarity base functions of CLIP (dashed line) and DINOv2 (solid line) with r ≥ 0.88.

### Major Features

- Download pre-trained weights on large datasets in batch
- Automatically generate training scripts
- Train and evaluate models with various specifications including architecture, weight and number of classes 
- Gather results of experiments with various specifications


## Installation
Locate to the target folder in your machine. Follow the instructions to install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

We provide the conda environment in ```fca.yaml```. Users can create the environment using this file by:
```
conda env create -f fca.yaml
```

Alternatively, users can choose to create the environment themselves. To do that, create a conda environment with ```Python3.8``` version:
```
conda create --name fca python=3.8 -y
conda activate fca
```
Build from this repository:
```
git clone https://github.com/fewclassarena/fca
cd fca
pip install -e .
pip install transformers
```
In case bugs occur, follow the instructions (in https://mmpretrain.readthedocs.io/en/latest/get_started.html#installation) to install [OpenMMLab](https://openmmlab.com/) (where ```Few-Class Arena``` is built upon) from scratch:
```
cd ..
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
pip install -U openmim && mim install -e .
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
mim install mmcv==2.1.0
```


## Versions

| Python | PyTorch | Conda | Pip | CUDA | GPU |
| -- | -- | -- | -- | -- | -- |
| 3.8.18 | 1.9.1 | 4.12.0 | 23.3.1 | cu111 | NVIDIA RTX A5000 |

The version information can be obtained by:
```
python3 -c 'import torch; print(torch.__version__); print(torch.version.cuda)'
```

Please refer to [installation documentation](https://mmpretrain.readthedocs.io/en/latest/get_started.html) for more detailed installation and dataset preparation.

## Repository Structure
The following scripts are newly designed and incorporated in the existing [MMPreTrain](https://openmmlab.com/) framework:
```
fca/
    ├── configs/
    │   ├── _base_/
    │   │   ├── datasets/
    │   │   │   ...
    │   │   ├── models/
    │   │   │   ...
    │   │   ├── schedules/
    │   │   ├── sim.py
    │   └── ...
    ├── datasets/
    │   ├── caltech101.py
    │   ├── caltech256.py
    │   ├── cifar100.py
    │   ├── cub200.py
    │   ├── ds.yaml
    │   ├── food101.py
    │   ├── gtsrb43.py
    │   ├── indoor67.py
    │   ├── sun397.py
    │   ├── textures47.py
    |   └── ...
    ├── dataset_converters/
    │   ├── convert_imagenet_ncls.py
    │   ├── convert_imagenet_noclsdir.py
    │   ├── convert_ncls.py
    │   └── ...
    ├── tools/
    │   ├── ncls/
    │   │   ├── config_to_url.yaml
    │   │   ├── download_weights.py
    │   │   ├── gen_configs.yaml
    │   │   ├── ncls_datasets_models_EDIT.yaml
    |   └── ...
    └── ...
```
The usage of each file will be illustrated in the following user guidelines.

## User Guidelines
Note that all scripts run in this main directory:
```
cd fca
```


### Generate configs for datasets and models
Specify ```meta_data_root``` in
```
./tools/ncls/datasets/ds.yaml
```
Download datasets:
```
python3 tools/ncls/datasets/<DATASET>.py
```
```FCA``` provides the following examples of scripts to download datasets:
```
python3 tools/ncls/datasets/caltech101.py
python3 tools/ncls/datasets/caltech256.py
python3 tools/ncls/datasets/cifar100.py
python3 tools/ncls/datasets/cub200.py
python3 tools/ncls/datasets/food101.py
python3 tools/ncls/datasets/gtsrb43.py
python3 tools/ncls/datasets/indoor67.py
python3 tools/ncls/datasets/sun397.py
python3 tools/ncls/datasets/textures47.py
```
For ```ImageNet1K```, please refer to [LSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/index.php#).

Specify datasets, models, and model EDIT files in ```./tools/ncls/ncls_datasets_models_EDIT.yaml``` in the following format:
```
datasets:
  - <DATASET>:
      ncls: <NUMBER_OF_CLASSES>
  ...

models:
  <MODEL>: <PATH_TO_MODEL_EDIT_FILE>
  ...
```
Example:
```
datasets:
  - imagenet:
      ncls: 1000
  - caltech101:
      ncls: 101

models:
  resnet50_8xb32_in1k: ./configs/resnet/resnet50_8xb32_in1k_EDIT.py
  vgg16_8xb32_in1k: ./configs/vgg/vgg16_8xb32_in1k_EDIT.py
```
A model EDIT file is a special base file in the ```FCA``` from which new configuration files are generated. This file defines the method for dataset, model, training and testing models and similarity such that downstream configuration files can be generated for specific experiments by a special marker ```# edit```, which helps the script to generate a list of files when varying the number of classes.

Then generate configs automatically by
```
python3 tools/ncls/gen_ncls_models_configs_EDIT.py
```


### Convert the dataset format
Specify ```meta_data_root``` where the meta data root is located in your current file system in ```./tools/ncls/datasets/ds.yaml``` in the following format:
```
meta_data_root: '<PATH_TO_DATASETS>'
```
The meta data root is a directory to store all your datasets.
Example:
```
meta_data_root: /datasets
```

The dataset format follows the convension of ImageNet:
```
imagenet1k/
    ├── meta
    │   ├── train.txt
    │   └── val.txt
    ├── train
    │   ├── <IMAGE_ID>.jpeg
    │   └── ...
    └── val
        ├── <IMAGE_ID>.jpeg
        └── ...
```
where a ```.txt``` file stores a pair of image id and and class number in each row in the following format
```
<IMAGE_ID>.jpeg <CLASS_NUM>
```
We follow the same ```train/val``` splits when the original dataset has already provided. If the dataset does not have explicit splits, we first assign image IDs to all images, starting from 0, and select ```4/5``` of all images as training set and put the rest in the validation set. Specifially, when an image whose ID satisfies the condition ```ID % 5 == 0```, it will be moved to the validation set. Otherwise, it will be assigned as a training sample.



Specify datasets, the number of classes in each full dataset, and the models and model EDIT files in ```./tools/ncls/ncls_datasets_models_EDIT.yaml``` in the following format:
```
datasets:
  - <DATASET>:
      ncls: <NUMBER_OF_CLASSES>
  ...

models:
  <MODEL>: <PATH_TO_MODEL_EDIT_FILE>
  ...
```
Example:

```
datasets:
  - imagenet:
      ncls: 1000
  - caltech101:
      ncls: 101

models:
  resnet50_8xb32_in1k: ./configs/resnet/resnet50_8xb32_in1k_EDIT.py
  vgg16_8xb32_in1k: ./configs/vgg/vgg16_8xb32_in1k_EDIT.py
```
Then run,
```
python3 tools/dataset_converters/convert_ncls.py
```


### Generate ncls meta files (for ImageNet1K)
Specify the number of classes by ```-ncls <NUM_OF_CLASSES>``` in ```tools/ncls/gen_ncls_meta_files.sh```. Replace ```<NUM_OF_CLASSES>``` with the number of classes. Note that one experiment with a specific ```ncls``` should be specified in one line. For example, if you would like to experiment with ```ncls=[2, 3, 4]```, then you would have three lines where each contains ```ncls 2```, ```ncls 3``` and ```ncls 4```, respectively.

```tools/ncls/gen_ncls_meta_files.sh``` already provides an example for ```ncls=[2, 3, 4, 5, 10, 100, 200, 400, 600, 800]```.

Then run
```
bash tools/ncls/gen_ncls_meta_files.sh
```


### Download pre-trained model weights
Specify the models and links to download in ```./tools/ncls/config_to_url.yaml``` in the following format:
```
<MODEL>: <LINK_OF_WEIGHTS>
```
where ```<MODEL>``` is replaced with the specific model and change ```<LINK_OF_WEIGHTS>``` to the link to download the weights.
Example:
```
resnet18_8xb32_in1k: https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth
```
Then run
```
python3 tools/ncls/download_weights.py
```


### FC-Full
```FC-Full``` evaluates models pre-trained on full datasets with the original number of classes (e.g. 1000 in ImageNet1K). Please refer to (#download-pre-trained-model-weights) regarding the details of downloading pre-trained weights from [MMPreTrain](https://openmmlab.com/).

Specify datasets, architectures, and models in the ```./tools/ncls/gen_configs.yaml``` in the following format:
```
datasets:
  - <DATASET>:
      ncls: <NUMBER_OF_CLASSES>
  ...

arch:
  <ARCHITECTURE>:
    path: <PATH_TO_ARCHITECTURE>
    model:
      - <PATH_TO_MODEL>
  ...
```
Example:
```
datasets:
  - imagenet:
      ncls: 1000
  - caltech101:
      ncls: 101

arch:
  resnet:
    path: ./configs/resnet
    model:
      - resnet18_8xb32_in1k
      - resnet34_8xb32_in1k
      - resnet50_8xb32_in1k
      - resnet101_8xb32_in1k
      - resnet152_8xb32_in1k
  vgg:
    path: ./configs/vgg
    model:
      - vgg16_8xb32_in1k
```
Generate configuration files and evaluate pre-trained models on full datasets
```
python3 tools/ncls/fc-full.py
```
Then results will be saved in ```./work_dirs/eval```.

For ```ImageNet1K``` use ```./tools/ncls/fc-full-IN1K.py```.

### FC-Sub
#### Training sub-models
```FC-Sub``` generates commands to train models on subsets with fewer classes. Note that the classes in the few-class subsets are randomly sampled from the full class using seed numbers. By default, we sample 5 subsets for each number of classes (ncls). The seed starts from ```0``` and will increment by 1 for each new subset.

Specify datasets, architectures and models in the ```./tools/ncls/gen_configs.yaml``` in the following format:
```
datasets:
  - <DATASET>:
      ncls: <NUMBER_OF_CLASSES>
  ...

arch:
  <ARCHITECTURE>:
    path: <PATH_TO_ARCHITECTURE>
    model:
      - <PATH_TO_MODEL>
  ...
```
Example:
```
datasets:
  - imagenet:
      ncls: 1000
  - caltech101:
      ncls: 101

arch:
  resnet:
    path: ./configs/resnet
    model:
      - resnet18_8xb32_in1k
      - resnet34_8xb32_in1k
      - resnet50_8xb32_in1k
      - resnet101_8xb32_in1k
      - resnet152_8xb32_in1k
  vgg:
    path: ./configs/vgg
    model:
      - vgg16_8xb32_in1k
```

Generate configs and train files scripts:
```
python3 tools/ncls/fc-sub.py
```
Then all training commands are generated in ```./tools/ncls/batch_train_<TIMESTAMP>.sh``` where ```<TIMESTAMP>``` will be specified by the script automatically.

If you have adequate hardware (e.g. GPUs) support to train all these models simultaneously, a single command would be enough:
```
bash ./tools/ncls/batch_train_<TIMESTAMP>.sh
```
However, this will easily get your server saturated. In practice, a user might want to have control of each model's training. One can simply view training scripts, and copy and paste each command in the command line:
```
vim ./tools/ncls/batch_train_<TIMESTAMP>.sh
```
Example of a ```./tools/ncls/batch_train_<TIMESTAMP>.sh``` file:
```
CUDA_VISIBLE_DEVICES=0 nohup python3 tools/train.py ./configs/resnet/resnet18_8xb32_in1k_ncls_2_s_0.py --amp > ./training_logs/resnet18_8xb32_in1k_ncls_2_s_0_2024_03_05_21:05:05.log & 
CUDA_VISIBLE_DEVICES=1 nohup python3 tools/train.py ./configs/resnet/resnet18_8xb32_in1k_ncls_2_s_1.py --amp > ./training_logs/resnet18_8xb32_in1k_ncls_2_s_1_2024_03_05_21:05:05.log & 
CUDA_VISIBLE_DEVICES=2 nohup python3 tools/train.py ./configs/resnet/resnet18_8xb32_in1k_ncls_2_s_2.py --amp > ./training_logs/resnet18_8xb32_in1k_ncls_2_s_2_2024_03_05_21:05:05.log &
...
```
where ```--amp``` enables ```automatic mixed precision``` training.

Each experiment is written in one line. Note that by ```nohup ... &```, the training will run in the background even if you log out. Training logs (optional) are written in ```./training_logs/*.log```. If errors occur due to the non-existence of the folder ```./training_logs```, you can simply create this folder by ```mkdir ./training_logs``` and execute the training scripts again. Another log files can be found in the experiment folder under the ```./work_dirs/``` path.

For ```ImageNet1K``` use ```./tools/ncls/fc-sub-IN1K.py```.

#### Fine-Tuning sub-models
```
python3 tools/train.py ./configs/<MODEL_CONFIG>.py --amp --resume <PATH_TO_FULL_MODEL_WEIGHTS>
```

#### Testing sub-models
Specify datasets, architectures, and models in the ```./tools/ncls/gen_configs.yaml``` in the format described in previous sections. Then run
```
python3 tools/ncls/fc-sub-res.py
```
which will search and evaluate the latest sub-models in ```./work_dirs```. Results will be saved in a log file with a timestamp under the ```./work_dirs/eval``` folder. Each line of the log file saves one evaluation result in this format: ```<DATASET_NAME>\t<MODEL>\t<NCLS>\t<SEED>\t<TOP1>\t<TOP5>\n```.

For ```ImageNet1K``` use ```./tools/ncls/fc-sub-res-IN1K.py```.

### FC-Sim
Specify datasets in ```./tools/ncls/ncls_datasets_models_EDIT.yaml``` in the following format:
```
datasets:
  - <DATASET>:
      ncls: <NUMBER_OF_CLASSES>
  ...
```
Example:
```
datasets:
  - imagenet:
      ncls: 1000
  - caltech101:
      ncls: 101
```
Specify the list of number of classes ```ncls_base_ls``` and ratios ```ncls_ratio``` in ```./tools/ncls/ncls_datasets_models_EDIT.yaml```. A complete list of number of classes will be the concatenation of ```ncls``` in ```ncls_base_ls``` and the final results of ```ncls_ratio``` multiplied by the number of classes in the full dataset (e.g. 10000 in ImageNet1K).

Example:
```
self.ncls_base_ls = [2, 3, 4, 5, 10]
self.ncls_ratio = [0.1, 0.2, 0.4, 0.6, 0.8]
```
Then a complete list of number of classes for ```ImageNet1K``` will be ```[2, 3, 4, 5, 10, 100, 200, 400, 600, 800]```. 

Users can specify the similarity base function by ```-sb <SIM_BASE_FUNCTION>``` or ```--sim_base <SIM_BASE_FUNCTION>``` when executing the ```./tools/ncls/fc-sim.py``` script. The similarity base function is defined in ```class Similarity``` in ```./configs/_base_/sim.py```

To use ```CLIP``` the similarity base function, run
```
python3 tools/ncls/fc-sim.py -sb CLIP
```
To use ```dinov2``` the similarity base function, run
```
python3 tools/ncls/fc-sim.py -sb dinov2
```
Results will be saved in a log file with a timestamp in ```./work_dirs/sim```, where each line saves results of one experiment in this format: ```<NCLS>\t<SEED>\t<S_ALPHA>\t<S_BETA>\t<S_SS>\n```.


## Contributing

We appreciate all contributions to improve Few-Class-Arena. Please fork this repository and make a pull request. We will review the changes and incorporate them into the existing code.


## Acknowledgement

Few-Class-Arena (FCA) is built upon the [MMPreTrain](https://openmmlab.com/) project. We thank the community for their invaluable contributions.

## Citation
```
@article{cao2024few,
  title={Few-Class Arena: A Benchmark for Efficient Selection of Vision Models and Dataset Difficulty Measurement},
  author={Cao, Bryan Bo and O'Gorman, Lawrence and Coss, Michael and Jain, Shubham},
  journal={arXiv preprint arXiv:2411.01099},
  year={2024}
}
```
```
@misc{fca2025,
  author = {Bryan Bo Cao},
  title = {Few-Class Arena (FCA)},
  year = {2025},
  howpublished = {\url{https://github.com/bryanbocao/fca}},
  note = {Accessed: 2025-03-01}
}
```
Note change the access date.

## License

This project is released under the [Apache 2.0 license](LICENSE).
