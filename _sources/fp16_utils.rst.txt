.. role:: hidden
    :class: hidden-section

apex.fp16_utils
===================================

This submodule contains utilities designed to streamline the mixed precision training recipe 
presented by NVIDIA `on Parallel Forall`_ and in GTC 2018 Sessions 
`Training Neural Networks with Mixed Precision: Theory and Practice`_ and 
`Training Neural Networks with Mixed Precision: Real Examples`_.
For Pytorch users, Real Examples in particular is recommended.

Full runnable Python scripts demonstrating ``apex.fp16_utils`` 
can be found on the Github page:

| `Simple FP16_Optimizer demos`_
|
| `Distributed Mixed Precision Training with imagenet`_
|
| `Mixed Precision Training with word_language_model`_
|
|

.. _`on Parallel Forall`:
    https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/
.. _`Training Neural Networks with Mixed Precision: Theory and Practice`:
    http://on-demand.gputechconf.com/gtc/2018/video/S8923/
.. _`Training Neural Networks with Mixed Precision: Real Examples`:
    http://on-demand.gputechconf.com/gtc/2018/video/S81012/
.. _`Simple FP16_Optimizer demos`:
    https://github.com/NVIDIA/apex/tree/master/examples/FP16_Optimizer_simple
.. _`Distributed Mixed Precision Training with imagenet`:
    https://github.com/NVIDIA/apex/tree/master/examples/imagenet
.. _`Mixed Precision Training with word_language_model`:
    https://github.com/NVIDIA/apex/tree/master/examples/word_language_model

.. automodule:: apex.fp16_utils
.. currentmodule:: apex.fp16_utils

Automatic management of master params + loss scaling
----------------------------------------------------

.. autoclass:: FP16_Optimizer
    :members:

.. autoclass:: LossScaler
    :members:

.. autoclass:: DynamicLossScaler
    :members:

Manual master parameter management
----------------------------------

.. autofunction:: prep_param_lists

.. autofunction:: master_params_to_model_params

.. autofunction:: model_grads_to_master_grads
