# Introduction to ASP

This page documents the API for ASP (Automatic Sparsity), a tool that enables sparse training and inference for PyTorch models by adding 2 lines of Python.

## Importing ASP
```
from apex.contrib.sparsity import ASP
```

## Initializing ASP

Apart from the import statement, it is sufficient to add just the following line of code before the training phase to augment the model and the optimizer for sparse training/infercence:
```
ASP.prune_trained_model(model, optimizer)
```

In a typical PyTorch training loop, it might look like this:

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
The `prune_trained_model` calculates the sparse mask and applies it to the weights. This is done once, i.e., sparse locations in the weights matrix remain fixed after this step. In order to recompute the sparse mask in between training, say after an epoch, use the following method:

```
ASP.compute_sparse_masks()
```

A more thorough example can be found in `./test/toy_problem.py`. 