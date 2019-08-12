#!/usr/bin/env python3

"""
This simple file provides an example of how to
 - import the pyprof library and initialize it
 - use the emit_nvtx context manager
 - start and stop the profiler

Only kernels within profiler.start and profiler.stop calls are profiled.
To profile
$ nvprof -f -o simple.sql --profile-from-start off ./simple.py
"""

import sys
import torch
import torch.cuda.profiler as profiler

#Import and initialize pyprof
from apex import pyprof
pyprof.nvtx.init()

a = torch.randn(5, 5).cuda()
b = torch.randn(5, 5).cuda()

#Context manager
with torch.autograd.profiler.emit_nvtx():

	#Start profiler
	profiler.start()

	c = a + b
	c = torch.mul(a,b)
	c = torch.matmul(a,b)
	c = torch.argmax(a, dim=1)
	c = torch.nn.functional.pad(a, (1,1))

	#Stop profiler
	profiler.stop()
