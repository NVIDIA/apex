#!/usr/bin/env python3

import torch
import torch.cuda.profiler as profiler
from apex import pyprof

class Foo(torch.nn.Module):
    def __init__(self, size):
        super(Foo, self).__init__()
        self.n = torch.nn.Parameter(torch.ones(size))
        self.m = torch.nn.Parameter(torch.ones(size))

    def forward(self, input):
        return self.n*input + self.m

foo = Foo(4)
foo.cuda()
x = torch.ones(4).cuda()

#JIT the class using tracing
traced_foo = torch.jit.trace(foo, x)

#Initialize pyprof after the JIT step
pyprof.nvtx.init()

#Assign a name to the object "traced_foo"
traced_foo.__dict__['__name__'] = "foo"

#Hook up the forward function to pyprof
pyprof.nvtx.wrap(traced_foo, 'forward')

with torch.autograd.profiler.emit_nvtx():
	profiler.start()
	z = traced_foo(x)
	profiler.stop()
	print(z)
