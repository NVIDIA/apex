# Mixed Precision DCGAN Training in PyTorch

This example is based off of pytorch/examples' dcgan example script and the commit referenced is https://github.com/pytorch/examples/blob/79d71b87d5bb46dc58da2dac5bf8289a7a2c3295/dcgan/main.py or older.

The differences are (a) this script can take a command line argument of `--dtype` to specify the dtype used during training and inference, and (b) this script uses CUDA by default.

Speaking about `--dtype` option, `torch.float32` is the default value.

`float16` has the script enable `torch.autocast`[^1] with device_type of CUDA and use `torch.cuda.amp.GradScaler`[^2].
`bfloat16` has the script enable `torch.autocast` with `torch.bfloat16`.

[^1]: https://pytorch.org/docs/stable/amp.html#torch.autocast
[^2]: https://pytorch.org/docs/stable/amp.html#gradient-scaling

Use `-h` or `--help` to see all the available command line options.
