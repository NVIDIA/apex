.. role:: hidden
    :class: hidden-section

apex.amp
===================================

Amp (Automatic Mixed Precision) is a tool designed for ease of use and maximum safety in FP16 training. All potentially unsafe ops are performed in FP32 under the hood, while safe ops are performed using faster, Tensor Core-friendly FP16 math. Amp also automatically implements dynamic loss scaling.

The intention of Amp is to be the "on-ramp" to easy FP16 training: achieve all the numerical stability of full FP32 training, with most of the performance benefits of full FP16 training.

Currently, complete API documentation resides on the Github page: https://github.com/NVIDIA/apex/tree/master/apex/amp.
