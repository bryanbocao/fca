python3 tools/dataset_converters/convert_imagenet_noclsdir.py -mp /datasets/imagenet/meta -ap train.txt

python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 2
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 3
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 4
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 5
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 10
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 100
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 200
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 400
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 600
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap train_noclsdir.txt -ncls 800

python3 tools/dataset_converters/convert_imagenet_noclsdir.py -mp /datasets/imagenet/meta -ap val.txt

python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 2
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 3
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 4
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 5
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 10
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 100
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 200
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 400
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 600
python3 tools/dataset_converters/convert_imagenet_ncls.py -mp /datasets/imagenet/meta -ap val_noclsdir.txt -ncls 800
