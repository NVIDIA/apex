#!/usr/bin/env python3

import torch
import torch.cuda.profiler as profiler
from apex import pyprof

#The following creates an object "foo" of type ScriptModule
#The new object has a function called "forward"

@torch.jit.script
def foo(x, y):
	return torch.sigmoid(x) + y

#Initialize pyprof after the JIT step
pyprof.nvtx.init()

#Assign a name to the object "foo"
foo.__name__ = "foo"

#Hook up the forward function to pyprof
pyprof.nvtx.wrap(foo, 'forward')

x = torch.zeros(4,4).cuda()
y = torch.ones(4,4).cuda()

with torch.autograd.profiler.emit_nvtx():
	profiler.start()
	z = foo(x, y)
	profiler.stop()
	print(z)
