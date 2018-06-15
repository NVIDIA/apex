# ImageNet training in PyTorch

This example is based on [https://github.com/pytorch/examples/tree/master/imagenet](https://github.com/pytorch/examples/tree/master/imagenet).
It implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset.

`main.py` and `main_fp16_optimizer.py` have been modified to use the `DistributedDataParallel` module in APEx instead of the one in upstream PyTorch.  For description of how this works please see the distributed example included in this repo.

`main.py` with the `--fp16` argument demonstrates mixed precision training with manual management of master parameters and loss scaling.

`main_fp16_optimizer.py` with `--fp16` demonstrates use of `apex.fp16_utils.FP16_Optimizer` to automatically manage master parameters and loss scaling.

To run multi-gpu on a single node use the command
```python -m apex.parallel.multiproc main.py ...```
adding any normal arguments.

## Requirements

- APEx which can be installed from https://www.github.com/nvidia/apex
- Install PyTorch from source, master branch of ([pytorch on github](https://www.github.com/pytorch/pytorch)
- `pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset.

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

```bash
python main.py -a alexnet --lr 0.01 /path/to/imagenet/folder
```

The directory at /path/to/imagenet/directory should contain two subdirectories called "train"
and "val" that contain the training and validation data respectively. Train images are expected to be 256x256 jpegs.

Example commands (note:  batch size --b 256 assumes your GPUs have >=16GB of onboard memory).

```bash
### Softlink training dataset into current directory
$ ln -sf /data/imagenet/train-jpeg-256x256/ train
### Softlink validation dataset into current directory
$ ln -sf /data/imagenet/val-jpeg/ val
### Single-process training
$ python main.py -a resnet50 --fp16 --b 256 --workers 4 ./
### Multi-process training (uses all visible GPU on the node)
$ python -m apex.parallel.multiproc main.py -a resnet50 --fp16 --b 256 --workers 4 ./
### Multi-process training on GPUs 0 and 1 only
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m apex.parallel.multiproc main.py -a resnet50 --fp16 --b 256 --workers 4 ./
### Multi-process training with FP16_Optimizer, default loss scale 1.0 (still uses FP32 master params)
$ python -m apex.parallel.multiproc main_fp16_optimizer.py -a resnet50 --fp16 --b 256 --workers 4 ./
# Multi-process training with FP16_Optimizer, static loss scale
$ python -m apex.parallel.multiproc main_fp16_optimizer.py -a resnet50 --fp16 --b 256 --static-loss-scale 128.0 --workers 4 ./
### Multi-process training with FP16_Optimizer, dynamic loss scaling
$ python -m apex.parallel.multiproc main_fp16_optimizer.py -a resnet50 --fp16 --b 256 --dynamic-loss-scale --workers 4 ./
```

## Usage for `main.py` and `main_fp16_optimizer.py`

```bash
usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained] [--fp16]
               [--static-loss-scale STATIC_LOSS_SCALE] [--prof]
               [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
               [--world-size WORLD_SIZE] [--rank RANK]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 | inception_v3
                        | resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --fp16                Run model fp16 mode.
  --static-loss-scale STATIC_LOSS_SCALE
                        Static loss scale, positive power of 2 values can improve
                        fp16 convergence.
  --prof                Only run 10 iterations for profiling.
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --world-size WORLD_SIZE
                        Number of GPUs to use. Can either be manually set or
                        automatically set by using 'python -m multiproc'.
  --rank RANK           Used for multi-process training. Can either be
                        manually set or automatically set by using 'python -m
                        multiproc'.
```

`main_fp16_optimizer.py` also accepts the optional flag
```bash
  --dynamic-loss-scale  Use dynamic loss scaling. If supplied, this argument
                        supersedes --static-loss-scale.
```

