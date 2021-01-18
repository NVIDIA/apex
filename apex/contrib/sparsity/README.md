# Introduction to ASP

This serves as a quick-start for ASP (Automatic SParsity), a tool that enables sparse training and inference for PyTorch models by adding 2 lines of Python.

## Importing ASP
```
from apex.contrib.sparsity import ASP
```

## Initializing ASP

Apart from the import statement, it is sufficient to add just the following line of code before the training phase to augment the model and the optimizer for sparse training/infercence:
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





