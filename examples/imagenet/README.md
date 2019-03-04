# Mixed Precision ImageNet Training in PyTorch

This example is based on [https://github.com/pytorch/examples/tree/master/imagenet](https://github.com/pytorch/examples/tree/master/imagenet).
It implements mixed precision training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset.

`main_amp.py` illustrates use of the new Amp API along with command-line flags (forwarded to `amp.initialize`) to easily manipulate and switch between various pure and mixed precision training modes.

Notice that with the new Amp API **you never need to explicitly convert your model, or the input data, to half().**

## Requirements

- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Training

To train a model, create softlinks to the Imagenet dataset, then run `main.py` with the desired model architecture, as shown in `Example commands` below.

The default learning rate schedule is set for ResNet50.  `main_amp.py` script rescales the learning rate according to the global batch size (number of distributed processes \* per-process minibatch size).

## Example commands

**Note:**  batch size `--b 224` assumes your GPUs have >=16GB of onboard memory.  You may be able to increase this to 256, but that's cutting it close, so it may out-of-memory for different Pytorch versions.

**Note:**  All of the following use 4 dataloader subprocesses (`--workers 4`) to reduce potential
CPU data loading bottlenecks.

**Note:**  `--opt-level` `O1` and `O2` both use dynamic loss scaling by default unless manually overridden.
`--opt-level` `O0` and `O3` (the "pure" training modes) do not use loss scaling by default.
`O0` and `O3` can be told to use loss scaling via manual overrides, but using loss scaling with `O0`
(pure FP32 training) does not really make sense, and will trigger a warning.

Softlink training and validation dataset into current directory
```
$ ln -sf /data/imagenet/train-jpeg/ train
$ ln -sf /data/imagenet/val-jpeg/ val
```

### Summary

Amp enables easy experimentation with various pure and mixed precision options.
```
$ python main_amp.py -a resnet50 --b 128 --workers 4 --opt-level O0 ./
$ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O3 ./
$ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O3 --keep-batchnorm-FP32 True ./
$ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O1 ./
$ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O1 --loss-scale 128.0 ./
$ python -m torch.distributed.launch --nproc_per_node=2 main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O1 ./
$ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O2 ./
$ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O2 --loss-scale 128.0 ./
$ python -m torch.distributed.launch --nproc_per_node=2 main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O2 ./
```
Options are broken down in detail below.

#### `--opt-level O0` (FP32 training) and `O3` (FP16 training)

"Pure FP32" training:
```
$ python main_amp.py -a resnet50 --b 128 --workers 4 --opt-level O0 ./
```
"Pure FP16" training:
```
$ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O3 ./
```
FP16 training with FP32 batchnorm:
```
$ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O3 --keep-batchnorm-FP32 True ./
```
Keeping the batchnorms in FP32 improves stability and allows Pytorch
to use cudnn batchnorms, which significantly increases speed in Resnet50.

The `O3` options might not converge, because they are not true mixed precision.
However, they can be useful to establish "speed of light" performance for
your model, which provides a baseline for comparison with `O1` and `O2`.
For Resnet50 in particular, `--opt-level O3 --keep-batchnorm-FP32 True` establishes
the "speed of light."  (Without `--keep-batchnorm-FP32`, it's slower, because it does
not use cudnn batchnorm.)

#### `--opt-level O1` ("conservative mixed precision")

`O1` patches Torch functions to cast inputs according to a whitelist-blacklist model.
FP16-friendly (Tensor Core) ops like gemms and convolutions run in FP16, while ops
that benefit from FP32, like batchnorm and softmax, run in FP32.
Also, dynamic loss scaling is used by default.
```
$ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O1 ./
```
`O1` overridden to use static loss scaling:
```
$ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O1 --loss-scale 128.0
```
Distributed training with 2 processes (1 GPU per process, see **Distributed training** below
for more detail)
```
$ python -m torch.distributed.launch --nproc_per_node=2 main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O1 ./
```
For best performance, set `--nproc_per_node` equal to the total number of GPUs on the node
to use all available resources.

#### `--opt-level O2` ("fast mixed precision")

`O2` casts the model to FP16, keeps batchnorms in FP32,
maintains master weights in FP32, and implements
dynamic loss scaling by default. (Unlike --opt-level O1, --opt-level O2
does not patch Torch functions.)
```
$ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O2 ./
```
"Fast mixed precision" overridden to use static loss scaling:
```
$ python main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O2 --loss-scale 128.0 ./
```
Distributed training with 2 processes (1 GPU per process)
```
$ python -m torch.distributed.launch --nproc_per_node=2 main_amp.py -a resnet50 --b 224 --workers 4 --opt-level O2 ./
```

## Distributed training

`main_amp.py` optionally uses `apex.parallel.DistributedDataParallel` (DDP) for multiprocess training with one GPU per process.
```
model = apex.parallel.DistributedDataParallel(model)
```
is a drop-in replacement for
```
model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[arg.local_rank],
                                                  output_device=arg.local_rank)
```
(because Torch DDP permits multiple GPUs per process, with Torch DDP you are required to
manually specify the device to run on and the output device.
With Apex DDP, it uses only the current device by default).

The choice of DDP wrapper (Torch or Apex) is orthogonal to the use of Amp and other Apex tools.  It is safe to use `apex.amp` with either `torch.nn.parallel.DistributedDataParallel` or `apex.parallel.DistributedDataParallel`.  In the future, I may add some features that permit optional tighter integration between `Amp` and `apex.parallel.DistributedDataParallel` for marginal performance benefits, but currently, there's no compelling reason to use Apex DDP versus Torch DDP for most models.

With both Apex DDP and Torch DDP, you must also call `torch.cuda.set_device(args.local_rank)` within
each process prior to initializing your model or any other tensors.
More information can be found in the docs for the
Pytorch multiprocess launcher module [torch.distributed.launch](https://pytorch.org/docs/stable/distributed.html#launch-utility).

`main_amp.py` is written to interact with 
[torch.distributed.launch](https://pytorch.org/docs/master/distributed.html#launch-utility),
which spawns multiprocess jobs using the following syntax:
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS main_amp.py args...
```
`NUM_GPUS` should be less than or equal to the number of visible GPU devices on the node.  The use of `torch.distributed.launch` is unrelated to the choice of DDP wrapper.  It is safe to use either apex DDP or torch DDP with `torch.distributed.launch`.

Optionally, one can run imagenet with synchronized batch normalization across processes by adding
`--sync_bn` to the `args...`

## Deterministic training (for debugging purposes)

Running with the `--deterministic` flag should produce bitwise identical outputs run-to-run,
regardless of what other options are used (see [Pytorch docs on reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)).
Since `--deterministic` disables `torch.backends.cudnn.benchmark`, `--deterministic` may
cause a modest performance decrease.
