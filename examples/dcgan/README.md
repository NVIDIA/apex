# Mixed Precision DCGAN Training in PyTorch

`main_amp.py` is based on [https://github.com/pytorch/examples/tree/master/dcgan](https://github.com/pytorch/examples/tree/master/dcgan).
It implements Automatic Mixed Precision (Amp) training of the DCGAN example for different datasets. Command-line flags forwarded to `amp.initialize` are used to easily manipulate and switch between various pure and mixed precision "optimization levels" or `opt_level`s.  For a detailed explanation of `opt_level`s, see the [updated API guide](https://nvidia.github.io/apex/amp.html).

We introduce these changes to the PyTorch DCGAN example as described in the [Multiple models/optimizers/losses](https://nvidia.github.io/apex/advanced.html#multiple-models-optimizers-losses) section of the documentation::
```
# Added after models and optimizers construction
[netD, netG], [optimizerD, optimizerG] = amp.initialize(
    [netD, netG], [optimizerD, optimizerG], opt_level=opt.opt_level, num_losses=3)
...
# loss.backward() changed to:
with amp.scale_loss(errD_real, optimizerD, loss_id=0) as errD_real_scaled:
    errD_real_scaled.backward()
...
with amp.scale_loss(errD_fake, optimizerD, loss_id=1) as errD_fake_scaled:
    errD_fake_scaled.backward()
...
with amp.scale_loss(errG, optimizerG, loss_id=2) as errG_scaled:
    errG_scaled.backward()
```

Note that we use different `loss_scalers` for each computed loss.
Using a separate loss scaler per loss is [optional, not required](https://nvidia.github.io/apex/advanced.html#optionally-have-amp-use-a-different-loss-scaler-per-loss).

To improve the numerical stability, we swapped `nn.Sigmoid() + nn.BCELoss()` to `nn.BCEWithLogitsLoss()`.

With the new Amp API **you never need to explicitly convert your model, or the input data, to half().**

"Pure FP32" training:
```
$ python main_amp.py --opt-level O0
```
Recommended mixed precision training:
```
$ python main_amp.py --opt-level O1
```

Have a look at the original [DCGAN example](https://github.com/pytorch/examples/tree/master/dcgan) for more information about the used arguments.

To enable mixed precision training, we introduce the `--opt-level` argument.
