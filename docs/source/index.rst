.. PyTorch documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://gitlab-master.nvidia.com/csarofeen/apex

APEx (A PyTorch Extension)
===================================

This is a repo is designed to hold PyTorch modules and utilities that are under active development and experimental. This repo is not designed as a long term solution or a production solution. Things placed in here are intended to be eventually moved to upstream PyTorch.

A major focus of this extension is the training of neural networks using 16-bit precision floating point math, which offers significant performance benefits on latest NVIDIA GPU architectures. The reduced dynamic range of half precision, however, is more vulnerable to numerical overflow/underflow.

APEX is an NVIDIA-maintained repository of utilities, including some that are targeted to improve the accuracy and stability of half precision networks, while maintaining high performance. The utilities are designed to be minimally invasive and easy to use.

Installation requires CUDA9, PyTorch 0.3 or later, and Python 3. Installation can be done by running
::
  git clone https://www.github.com/nvidia/apex
  cd apex
  python setup.py install

	       

.. toctree::
   :maxdepth: 1
   :caption: apex

   parallel
   reparameterization
   RNN
   fp16_utils
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
