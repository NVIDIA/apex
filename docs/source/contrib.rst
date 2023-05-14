.. module:: apex.contrib

apex.contrib
============

Bottleneck
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.bottleneck.Bottleneck
   apex.contrib.bottleneck.SpatialBottleneck
   apex.contrib.bottleneck.HaloExchangerNoComm
   apex.contrib.bottleneck.HaloExchangerAllGather
   apex.contrib.bottleneck.HaloExchangerSendRecv
   apex.contrib.bottleneck.HaloExchangerPeer


Clip Grad
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.clip_grad.clip_grad_norm_


cuDNN frontend based Conv-Bias-ReLU
-----------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.conv_bias_relu.ConvBiasReLU
   apex.contrib.conv_bias_relu.ConvBias
   apex.contrib.conv_bias_relu.ConvBiasMaskReLU


cuDNN based 2D Group Batch Normalization
----------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.cudnn_gbn.GroupBatchNorm2d


Fused MultiHead Attention
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.fmha.fmha.FMHA


Focal Loss
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.focal_loss.focal_loss.focal_loss


Group Batch Normalization
-------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.groupbn.BatchNorm2d_NHWC


2D Index Multiply
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.index_mul_2d.index_mul_2d


Layer Normalization
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.layer_norm.FastLayerNorm


MultiHead Attention
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.multihead_attn.SelfMultiheadAttn
   apex.contrib.multihead_attn.EncdecMultiheadAttn
   apex.contrib.multihead_attn.fast_mask_softmax_dropout_func


Optimizers
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.optimizers.distributed_fused_adam.DistributedFusedAdam
   apex.contrib.optimizers.distributed_fused_lamb.DistributedFusedLAMB


Peer Memory
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.peer_memory.PeerMemoryPool
   apex.contrib.peer_memory.PeerHaloExchanger1d


Sparsity
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.sparsity.create_mask
   apex.contrib.sparsity.ASP


Transducer
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.transducer.TransducerJoint
   apex.contrib.transducer.TransducerLoss


Cross Entropy
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   apex.contrib.xentropy.SoftmaxCrossEntropyLoss
