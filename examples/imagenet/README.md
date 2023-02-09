# Mixed Precision ImageNet Training in PyTorch

`main_amp.py` is based on [https://github.com/pytorch/examples/tree/master/imagenet](https://github.com/pytorch/examples/tree/master/imagenet).
It implements Automatic Mixed Precision (Amp) training of popular model architectures, such as ResNet, AlexNet, and VGG, on the ImageNet dataset.  Command-line flags forwarded to `torch.cuda.amp.autocast` are used to easily manipulate and switch between various pure and mixed precision.

Three lines enable Amp:
```python
# Enclose `model.forward` and loss computation with `autocast` context
with torch.autocast(device_type="cuda", dtype=dtype, enable=enable):
    pred = model(inputs)
    loss = loss_fn(target, pred)

...
# If fp16 is selected for AMP...
grad_scaler.scale(loss).backward()
grad_scaler.step(optimizer)
grad_scaler.update()
```

## Requirements

- Download the ImageNet dataset and move validation images to labeled subfolders
    - The following script may be helpful: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

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

Softlink training and validation datasets into the current directory:
```
$ ln -sf /data/imagenet/train-jpeg/ train
$ ln -sf /data/imagenet/val-jpeg/ val
```

### Summary

Amp allows easy experimentation with various pure and mixed precision options.
```shell
$ python main_amp.py -a resnet50 --b 128 --workers 4 ./
$ python main_amp.py -a resnet50 --b 224 --workers 4 ./
$ python main_amp.py -a resnet50 --b 224 --workers 4 ./
$ python main_amp.py -a resnet50 --b 224 --workers 4 ./
$ python main_amp.py -a resnet50 --b 224 --workers 4 ./
$ python -m torch.distributed.launch --nproc_per_node=2 main_amp.py -a resnet50 --b 224 --workers 4 ./
$ python main_amp.py -a resnet50 --b 224 --workers 4 ./
$ python main_amp.py -a resnet50 --b 224 --workers 4 ./
$ python -m torch.distributed.launch --nproc_per_node=2 main_amp.py -a resnet50 --b 224 --workers 4 ./
```


Distributed training with 2 processes (1 GPU per process, see **Distributed training** below
for more detail)
```
$ torchrun --nproc_per_node=2 main_amp.py -a resnet50 --b 224 --workers 4 ./
```
For best performance, set `--nproc_per_node` equal to the total number of GPUs on the node
to use all available resources.

## Distributed training

`main_amp.py` optionally uses `torch.nn.parallel.DistributedDataParallel` (DDP) for multiprocess training with one GPU per process.
```python
model = torch.nn.parallel.DistributedDataParallel(model)
```
is a drop-in replacement for

More information can be found in the docs for the
Pytorch multiprocess launcher module [torch.distributed.launch](https://pytorch.org/docs/stable/distributed.html#launch-utility).

`main_amp.py` is written to interact with
[torch.distributed.launch](https://pytorch.org/docs/master/distributed.html#launch-utility),
which spawns multiprocess jobs using the following syntax:
```shell
$ torchrun --nproc_per_node=NUM_GPUS main_amp.py args...
```
`NUM_GPUS` should be less than or equal to the number of visible GPU devices on the node.

Optionally, one can run imagenet with synchronized batch normalization across processes by adding
`--sync_bn` to the `args...`

## Deterministic training (for debugging purposes)

Running with the `--deterministic` flag should produce bitwise identical outputs run-to-run,
regardless of what other options are used (see [Pytorch docs on reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)).
Since `--deterministic` disables `torch.backends.cudnn.benchmark`, `--deterministic` may
cause a modest performance decrease.

## Profiling

If you're curious how the network actually looks on the CPU and GPU timelines (for example, how good is the overall utilization?
Is the prefetcher really overlapping data transfers?) try profiling `main_amp.py`.

See [
Using Nsight Systems to profile GPU workload (PyTorch Dev Discussions)](https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59) and [User Guide :: Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
