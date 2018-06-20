fp16_optimizer.py contains `FP16_Optimizer`, a Python class designed to wrap an existing Pytorch optimizer and automatically enable master parameters and loss scaling in a manner transparent to the user.  To use `FP16_Optimizer`, only two lines of one's Python model need to change.

#### [FP16_Optimizer API documentation](https://nvidia.github.io/apex/fp16_utils.html#automatic-management-of-master-params-loss-scaling)

#### [Simple examples with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/FP16_Optimizer_simple)

#### [Imagenet with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

#### [word_language_model with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/word_language_model)


fp16_util.py contains a number of utilities to manually manage master parameters and loss scaling, if the user chooses.  

#### [Manual management documentation](https://nvidia.github.io/apex/fp16_utils.html#manual-master-parameter-management)

The [Imagenet with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/imagenet) and [word_language_model with FP16_Optimizer](https://github.com/NVIDIA/apex/tree/master/examples/word_language_model) directories also contain `main.py` files that demonstrate manual management of master parameters and static loss scaling.  These examples illustrate what sort of operations `FP16_Optimizer` is performing automatically.
