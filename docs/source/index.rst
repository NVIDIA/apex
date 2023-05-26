.. PyTorch documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/nvidia/apex

Apex (A PyTorch Extension)
===================================

This site contains the API documentation for Apex (https://github.com/nvidia/apex),
a Pytorch extension with NVIDIA-maintained utilities to streamline mixed precision and distributed training.  Some of the code here will be included in upstream Pytorch eventually. The intention of Apex is to make up-to-date utilities available to users as quickly as possible.

Installation instructions can be found here:  https://github.com/NVIDIA/apex#quick-start.

Some other useful material, including GTC 2019 and Pytorch DevCon 2019 Slides, can be found here:  https://github.com/mcarilli/mixed_precision_references.

.. toctree::
   :maxdepth: 1
   :caption: AMP:  Automatic Mixed Precision

   amp

.. toctree::
   :maxdepth: 1
   :caption: Distributed Training

   parallel

.. toctree::
   :maxdepth: 1
   :caption: Fused Optimizers

   optimizers

.. toctree::
   :maxdepth: 1
   :caption: Fused Layer Norm

   layernorm

..   .. toctree::
     :maxdepth: 1
     :caption: Deprecated mixed precision API
     fp16_util

..   RNN
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
