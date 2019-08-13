#!/usr/bin/env python3

"""
This file checks all Python operators.
"""

import sys
import torch
import torch.cuda.profiler as profiler
import operator
import inspect

#Import and initialize pyprof
from apex import pyprof
pyprof.nvtx.init()

X = 1024
Y = 1024

fa = torch.rand(X, Y).cuda()
fb = torch.rand(X, Y).cuda()
fc = torch.rand(X, Y).cuda()

ia = torch.randint(0, 100, (X, Y)).cuda()
ib = torch.randint(0, 100, (X, Y)).cuda()

sa = torch.ones(1,1).cuda()
sb = torch.ones(1,1).cuda()

ba = fa.byte()

unaryOps = ["abs", "__abs__", "neg", "__neg__",]
invertOps = ["inv", "invert", "__inv__", "__invert__",]	#imlemented only for byte tensors
#pos, __pos__ is not implemented for tensors

binaryOps = []
binaryOps += [ "lt", "__lt__", "le", "__le__", "eq", "__eq__", "ne", "__ne__", "ge", "__ge__", "gt", "__gt__" ]
binaryOps += [ "add", "__add__", "sub", "__sub__", "mul", "__mul__", "floordiv", "__floordiv__", "truediv", "__truediv__", "pow", "__pow__", "mod", "__mod__"]
binaryOps += [ "and_", "__and__", "or_", "__or__", "xor", "__xor__", "lshift", "__lshift__", "rshift", "__rshift__"]

inplaceOps = []
inplaceOps += ["iadd", "__iadd__", "isub", "__isub__", "imul", "__imul__", "ifloordiv", "__ifloordiv__", "itruediv", "__itruediv__", "imod", "__imod__",]
#ipow, __ipow__ is not implemented in pytorch
inplaceOps += [ "iand", "__iand__", "ior", "__ior__", "ixor", "__ixor__", "ilshift", "__ilshift__", "irshift", "__irshift__",]

matmulOps = [ "matmul", "__matmul__" ]
inplacematmulOps = [ "imatmul", "__imatmul__" ]

reverseIntBinaryOps = ["__radd__", "__rsub__", "__rmul__", "__rfloordiv__", "__rpow__",]
reverseFloatBinaryOps = ["__radd__", "__rsub__", "__rmul__", "__rdiv__", "__rtruediv__", "__rfloordiv__", "__rpow__",]

'''
TODO
.concat(a, b)
.__concat__(a, b)
.contains(a, b)
.__contains__(a, b)
.countOf(a, b)
.delitem(a, b)
.__delitem__(a, b)
.getitem(a, b)
.__getitem__(a, b)
.indexOf(a, b)
.setitem(a, b, c)
.__setitem__(a, b, c)
.length_hint(obj, default=0)
.iconcat(a, b)
.__iconcat__(a, b)
.index(a)
.__index__(a)
'''

#Context manager
with torch.autograd.profiler.emit_nvtx():

	#Start profiler
	profiler.start()

	for op in unaryOps:
		assert hasattr(operator, op)
		f = getattr(operator, op)
		assert inspect.isbuiltin(f)
		c = f(ia)

	for op in invertOps:
		assert hasattr(operator, op)
		f = getattr(operator, op)
		assert inspect.isbuiltin(f)
		c = f(ba)

	for op in binaryOps:
		assert hasattr(operator, op)
		f = getattr(operator, op)
		assert inspect.isbuiltin(f)
		c = f(ia, ib)
		c = f(ia, 2)

	for op in inplaceOps:
		assert hasattr(operator, op)
		f = getattr(operator, op)
		assert inspect.isbuiltin(f)
		ia = f(ia, ib)
		ia = f(ia, 2)

	for op in matmulOps:
		assert hasattr(operator, op)
		f = getattr(operator, op)
		assert inspect.isbuiltin(f)
		c = f(fa, fb)

	for op in inplacematmulOps:
		assert hasattr(operator, op)
		f = getattr(operator, op)
		assert inspect.isbuiltin(f)
		fa = f(fa, fb)

	for op in reverseIntBinaryOps:
		assert hasattr(torch.Tensor, op)
		f = getattr(torch.Tensor, op)
		ia = f(ia, ib)

	for op in reverseFloatBinaryOps:
		assert hasattr(torch.Tensor, op)
		f = getattr(torch.Tensor, op)
		fa = f(fa, fb)

	'''
	#c = fa[3]
	#c = fa[3][3]
	#c = torch.min(fa, 3)
	c = torch.sum(fa)
	c = torch.max(fa)
	c = -fa
	#fc[2][2] = fa[2][2]

	c = a_scalar and b_scalar
	c = a_scalar or b_scalar
	c = not a_scalar

	c = a is b
	c = a is not b
	'''

	#Stop profiler
	profiler.stop()
