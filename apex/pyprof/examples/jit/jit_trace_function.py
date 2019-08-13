#!/usr/bin/env python3

import torch
import torch.cuda.profiler as profiler
from apex import pyprof

def foo(x, y):
	return torch.sigmoid(x) + y

x = torch.zeros(4,4).cuda()
y = torch.ones(4,4).cuda()

#JIT the function using tracing
#This returns an object of type ScriptModule with a forward method.
traced_foo = torch.jit.trace(foo, (x,y))

#Initialize pyprof after the JIT step
pyprof.nvtx.init()

#Assign a name to the object "traced_foo"
traced_foo.__dict__['__name__'] = "foo"

#Hook up the forward function to pyprof
pyprof.nvtx.wrap(traced_foo, 'forward')

with torch.autograd.profiler.emit_nvtx():
	profiler.start()
	z = traced_foo(x, y)
	profiler.stop()
	print(z)
