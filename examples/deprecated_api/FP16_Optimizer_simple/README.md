# Simple examples of FP16_Optimizer functionality

To use `FP16_Optimizer` on a half-precision model, or a model with a mixture of 
half and float parameters, only two lines of your training script need to change:
1. Construct an `FP16_Optimizer` instance from an existing optimizer.
2. Replace `loss.backward()` with `optimizer.backward(loss)`.

#### [Full API Documentation](https://nvidia.github.io/apex/fp16_utils.html#automatic-management-of-master-params-loss-scaling)

See "Other Options" at the bottom of this page for some cases that require special treatment.

#### Minimal Working Sample
`minimal.py` shows the basic usage of `FP16_Optimizer` with either static or dynamic loss scaling.  Test via `python minimal.py`.

#### Closures
`FP16_Optimizer` supports closures with the same control flow as ordinary Pytorch optimizers.  
`closure.py` shows an example.  Test via `python closure.py`.

See [the API documentation](https://nvidia.github.io/apex/fp16_utils.html#apex.fp16_utils.FP16_Optimizer.step) for more details.

#### Serialization/Deserialization
`FP16_Optimizer` supports saving and loading with the same control flow as ordinary Pytorch optimizers.
`save_load.py` shows an example.  Test via `python save_load.py`.

See [the API documentation](https://nvidia.github.io/apex/fp16_utils.html#apex.fp16_utils.FP16_Optimizer.load_state_dict) for more details.

#### Distributed
**distributed_apex** shows an example using `FP16_Optimizer` with Apex DistributedDataParallel.
The usage of `FP16_Optimizer` with distributed does not need to change from ordinary single-process 
usage. Test via
```bash
cd distributed_apex
bash run.sh
```

**distributed_pytorch** shows an example using `FP16_Optimizer` with Pytorch DistributedDataParallel.
Again, the usage of `FP16_Optimizer` with distributed does not need to change from ordinary 
single-process usage.  Test via
```bash
cd distributed_pytorch
bash run.sh
```

#### Other Options

Gradient clipping requires that calls to `torch.nn.utils.clip_grad_norm`
be replaced with [fp16_optimizer_instance.clip_master_grads()](https://nvidia.github.io/apex/fp16_utils.html#apex.fp16_utils.FP16_Optimizer.clip_master_grads).  The [word_language_model example](https://github.com/NVIDIA/apex/blob/master/examples/word_language_model/main_fp16_optimizer.py) uses this feature.

Multiple losses will work if you simply replace
```bash
loss1.backward()
loss2.backward()
```
with 
```bash
optimizer.backward(loss1)
optimizer.backward(loss2)
```
but `FP16_Optimizer` can be told to handle this more efficiently using the 
[update_master_grads()](https://nvidia.github.io/apex/fp16_utils.html#apex.fp16_utils.FP16_Optimizer.update_master_grads) option.
