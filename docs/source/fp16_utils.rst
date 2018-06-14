.. role:: hidden
    :class: hidden-section

apex.fp16_utils
===================================

This submodule contains utilities designed to streamline the mixed precision training recipe 
presented by NVIDIA `on Parallel Forall`_ and in GTC 2018 Sessions 
`Training Neural Networks with Mixed Precision: Theory and Practice`_ and 
`Training Neural Networks with Mixed Precision: Real Examples`_.
For Pytorch users, Real Examples in particular is recommended.

.. _`on Parallel Forall`:
    https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/
.. _`Training Neural Networks with Mixed Precision: Theory and Practice`:
    http://on-demand.gputechconf.com/gtc/2018/video/S8923/
.. _`Training Neural Networks with Mixed Precision: Real Examples`:
    http://on-demand.gputechconf.com/gtc/2018/video/S81012/

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


Custom Operations
-----------------

.. autoclass:: Fused_Weight_Norm
    :members:
