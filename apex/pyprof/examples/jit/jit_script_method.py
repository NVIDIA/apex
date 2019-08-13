#!/usr/bin/env python3

import torch
import torch.cuda.profiler as profiler
from apex import pyprof

class Foo(torch.jit.ScriptModule):
    def __init__(self, size):
        super(Foo, self).__init__()
        self.n = torch.nn.Parameter(torch.ones(size))
        self.m = torch.nn.Parameter(torch.ones(size))

    @torch.jit.script_method
    def forward(self, input):
        return self.n*input + self.m

#Initialize pyprof after the JIT step
pyprof.nvtx.init()

#Hook up the forward function to pyprof
pyprof.nvtx.wrap(Foo, 'forward')

foo = Foo(4)
foo.cuda()
x = torch.ones(4).cuda()

with torch.autograd.profiler.emit_nvtx():
	profiler.start()
	z = foo(x)
	profiler.stop()
	print(z)
