.. PyTorch documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://gitlab-master.nvidia.com/csarofeen/apex

Pychu (Pytorch Cuda Half Utilities)
===================================

Training neural networks using 16-bit half precision floating point
(as opposed to traditional 32-bit floating point)
offers significant performance benefits
on the latest NVIDIA Volta GPUs.  
However, the reduced dynamic range of half precision is more
vulnerable to numerical overflow/underflow.

Pychu is an NVIDIA-maintained repository of utilities that improve the
accuracy and stability of half precision networks, while maintaining high performance.
The utilities are designed to be minimally invasive and easy to use.

If you've got a working Pytorch implementation of your network 
that uses 32-bit floating point, 
you should be able to realize immediate performance gains of 2X or more 
after changing only a few lines of code.

.. toctree::
   :maxdepth: 1
   :caption: apex

   parallel
   reparameterization
   RNN
   utils
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
