# Introduction to ASP

This serves as a quick-start for ASP (Automatic SParsity), a tool that enables sparse training and inference for PyTorch models by adding 2 lines of Python.

## Importing ASP

```
from apex.contrib.sparsity import ASP
```

## Initializing ASP

Apart from the import statement, it is sufficient to add just the following line of code before the training phase to augment the model and the optimizer for sparse training/inference:

```
ASP.prune_trained_model(model, optimizer)
```

In the context of a typical PyTorch training loop, it might look like this:

```
ASP.prune_trained_model(model, optimizer)

x, y = DataLoader(args)
for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_function(y_pred, y)
    loss.backward()
    optimizer.step()

torch.save(...)
```

The `prune_trained_model` step calculates the sparse mask and applies it to the weights. This is done once, i.e., sparse locations in the weights matrix remain fixed after this step. 

## Generate a Sparse Network

The following approach serves as a guiding example on how to generate a pruned model that can use Sparse Tensor Cores in the NVIDIA Ampere Architecture. This approach generates a model for deployment, i.e. inference mode.

```
(1) Given a fully trained (dense) network, prune parameter values in a 2:4 sparse pattern.
(2) Fine-tune  the  pruned  model  with  optimization  method  and  hyper-parameters (learning-rate, schedule, number of epochs, etc.) exactly as those used to obtain the trained model.
(3) (If required) Quantize the model.
```

In code, below is a sketch on how to use ASP for this approach (steps 1 and 2 above).

```
model = define_model(..., pretrained=True) # define model architecture and load parameter tensors with trained values (by reading a trained checkpoint)
criterion = ... # compare ground truth with model predition; use the same criterion as used to generate the dense trained model
optimizer = ... # optimize model parameters; use the same optimizer as used to generate the dense trained model
lr_scheduler = ... # learning rate scheduler; use the same schedule as used to generate the dense trained model

from apex.contrib.sparsity import ASP     
ASP.prune_trained_model(model, optimizer) #pruned a trained model

x, y = DataLoader(args)
for epoch in range(epochs): # train the pruned model for the same number of epochs as used to generate the dense trained model
    y_pred = model(x)
    loss = criterion(y_pred, y)
    lr_scheduler.step()
    loss.backward()
    optimizer.step()

torch.save(...) # saves the pruned checkpoint with sparsity masks 
```

## Non-Standard Usage

If your goal is to easily perpare a network for accelerated inference, please follow the recipe above.  However, ASP can also be used to perform experiments in advanced techniques like training with sparsity from initialization. For example, in order to recompute the sparse mask in between training steps, use the following method:

```
ASP.compute_sparse_masks()
```

A more thorough example can be found in `./test/toy_problem.py`. 

## Advanced Usage: Channel Permutation

We introduce channel permutations as an advanced method to maximize the accuracy of structured sparse networks. By permuting weight matrices along their channel dimension and adjusting the surrounding layers appropriately, we demonstrate accuracy recovery for even small, parameter-efficient networks, without affecting inference run-time.

The final accuracy has a strong relationship with the quality of permutations. We provide the default algorithms to search for high-quality permutations. The permutation search process can be accelerated by the Apex CUDA extension: `apex.contrib.sparsity.permutation_search_kernels`

If you want to use the GPU to accelerate the permutation search process, we recommend installing Apex with permutation search CUDA extension via

```
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--permutation_search" ./
```

If you want to disable the permutation search process, please pass the `allow_permutation=False` to `init_model_for_pruning` function. For example:

```
ASP.init_model_for_pruning(model, mask_calculator="m4n2_1d", verbosity=2, whitelist=[torch.nn.Linear, torch.nn.Conv2d], allow_recompute_mask=False, allow_permutation=False)
```

Please notice, when using multi-GPUs we should set the identical random seed for all GPUs to make sure the same results generated in permutation search. The library has implemented the `set_identical_seed` function in `permutation_lib.py`, and be called in ASP library. We still suggest the users to set the identical random seed when using multi-GPUs in their code, the example code is as follows:

```
import torch
import numpy
import random

torch.manual_seed(identical_seed)
torch.cuda.manual_seed_all(identical_seed)
numpy.random.seed(identical_seed)
random.seed(identical_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Reference Papers

More details about sparsity support on the NVIDIA Ampere GPU with Sparse Tensor Cores can refer to our [white paper](https://arxiv.org/abs/2104.08378).

```
@article{mishra2021accelerating,
  title={Accelerating sparse deep neural networks},
  author={Mishra, Asit and Latorre, Jorge Albericio and Pool, Jeff and Stosic, Darko and Stosic, Dusan and Venkatesh, Ganesh and Yu, Chong and Micikevicius, Paulius},
  journal={arXiv preprint arXiv:2104.08378},
  year={2021}
}
```

The details about sparsity with permutation can refer to our [paper](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html) published in *Thirty-fifth Conference on Neural Information Processing Systems* (**NeurIPS 2021**):

```
@article{pool2021channel,
  title={Channel Permutations for N: M Sparsity},
  author={Pool, Jeff and Yu, Chong},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
