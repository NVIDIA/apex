# ImageNet training in PyTorch

This example is based on [https://github.com/pytorch/examples/tree/master/imagenet](https://github.com/pytorch/examples/tree/master/imagenet).
It implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset.

`main.py` with the `--fp16` argument demonstrates mixed precision training with manual management of master parameters and loss scaling.

`main_fp16_optimizer.py` with `--fp16` demonstrates use of `apex.fp16_utils.FP16_Optimizer` to automatically manage master parameters and loss scaling.

`apex.parallel.DistributedDataParallel` automatically allreduces and averages gradients during `backward()`.  If you wish to control the allreduce manually instead (for example, to carry out the allreduce every few iterations instead of every iteration), [apex.parallel.reduce](https://nvidia.github.io/apex/parallel.html#apex.parallel.Reducer) provides a convenient wrapper.  
`main_reducer.py` is identical to `main.py`, except that it shows the use of `Reducer` instead of `DistributedDataParallel`.

## Requirements

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

## Distributed training

`main.py` and `main_fp16_optimizer.py` have been modified to use the `DistributedDataParallel` module in Apex instead of the one in upstream PyTorch. `apex.parallel.DistributedDataParallel` 
is a drop-in replacement for `torch.nn.parallel.DistribtuedDataParallel` (see our [distributed example](https://github.com/NVIDIA/apex/tree/master/examples/distributed)).  
The scripts can interact with 
[torch.distributed.launch](https://pytorch.org/docs/master/distributed.html#launch-utility)
to spawn multiprocess jobs using the following syntax:
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS main.py args...
```
`NUM_GPUS` should be less than or equal to the number of visible GPU devices on the node.

## Example commands

(note:  batch size `--b 256` assumes your GPUs have >=16GB of onboard memory)

```bash
### Softlink training dataset into current directory
$ ln -sf /data/imagenet/train-jpeg-256x256/ train
### Softlink validation dataset into current directory
$ ln -sf /data/imagenet/val-jpeg/ val
### Single-process training
$ python main.py -a resnet50 --fp16 --b 256 --workers 4 ./
### Multi-process training (uses all visible GPU on the node)
$ python -m torch.distributed.launch --nproc_per_node=NUM_GPUS main.py -a resnet50 --fp16 --b 256 --workers 4 ./
### Multi-process training on GPUs 0 and 1 only
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m torch.distributed.launch --nproc_per_node=2 main.py -a resnet50 --fp16 --b 256 --workers 4 ./
### Multi-process training with FP16_Optimizer, default loss scale 1.0 (still uses FP32 master params)
$ python -m torch.distributed.launch --nproc_per_node=NUM_GPUS main_fp16_optimizer.py -a resnet50 --fp16 --b 256 --workers 4 ./
# Multi-process training with FP16_Optimizer, static loss scale
$ python -m torch.distributed.launch --nproc_per_node=NUM_GPUS main_fp16_optimizer.py -a resnet50 --fp16 --b 256 --static-loss-scale 128.0 --workers 4 ./
### Multi-process training with FP16_Optimizer, dynamic loss scaling
$ python -m torch.distributed.launch --nproc_per_node=NUM_GPUS main_fp16_optimizer.py -a resnet50 --fp16 --b 256 --dynamic-loss-scale --workers 4 ./
```

## Usage for `main.py` and `main_fp16_optimizer.py`

`main_fp16_optimizer.py` also accepts the optional flag
```bash
  --dynamic-loss-scale  Use dynamic loss scaling. If supplied, this argument
                        supersedes --static-loss-scale.
```

