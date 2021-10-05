# apex.transformer

`apex.transformer` is a module which enables efficient large Transformer models at scale.

`apex.transformer.tensor_parallel` and `apex.transformer.pipeline_parallel` are both based on [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)'s module.
The former is based on `megatron.mpu` and the latter is on `megatron.schedules` and `megatron.p2p_communication`.
