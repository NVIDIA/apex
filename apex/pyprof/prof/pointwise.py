import numpy as np
from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Pointwise(OperatorLayerBase):

	ops = []
	ops += ["__abs__", "__neg__", "__invert__"]
	ops += ["__add__", "__sub__", "__mul__", "__floordiv__", "__truediv__", "__pow__", "__mod__"]
	ops += ["__radd__", "__rsub__", "__rmul__", "__rdiv__", "__rtruediv__", "__rfloordiv__", "__rpow__"]
	ops += ["__iadd__", "__isub__", "__imul__", "__itruediv__",]
	ops += ["__lt__", "__gt__", "__ge__", "__le__", "__eq__", "__ne__",]
	ops += ["lt", "lt_", "gt", "gt_", "ge", "ge_", "le", "le_", "eq", "eq_", "ne", "ne_",]
	ops += ["__and__", "__or__", "__xor__", "__lshift__", "__rshift__"]
	ops += ["__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__"]
	ops += ["abs", "abs_", "neg", "neg_"]
	ops += ["add", "add_", "div", "div_", "mul", "mul_", "reciprocal", "reciprocal_", "remainder", "remainder_", "sub", "sub_",]
	ops += ["addcdiv", "addcdiv_", "addcmul", "addcmul_"]
	ops += ["exp", "exp_", "exp1m", "exp1m_", "log", "log_", "log10", "log10_", "log1p", "log1p_", "log2", "log2_", "pow", "pow_", "rsqrt", "rsqrt_", "sqrt", "sqrt_",]
	ops += ["ceil", "ceil_", "clamp", "clamp_", "floor", "floor_", "fmod", "fmod_", "frac", "frac_", "round", "round_", "sign", "sign_", "trunc", "trunc_"]
	ops += ["acos", "acos_", "asin", "asin_", "atan", "atan_", "atan2", "atan2_", "cos", "cos_", "cosh", "cosh_", "sin", "sin_", "sinh", "sinh_", "tan", "tan_", "sigmoid", "sigmoid_", "tanh", "tanh_"]
	ops += ["digamma", "erf", "erf_", "erfc", "erfc_", "erfinv", "erfinv_", "lerp", "lerp_", "mvlgamma",]

	@staticmethod
	def foo(d):
		return d['name'],d['type'],d['shape'],d['dtype']

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		self.dir = d.dir
		assert (d.dir in ["fprop", "bprop"])
		assert (op in Pointwise.ops)

		#Filter out all named parameters (kwargs).
		#This might require revisiting in future.
		args = list(filter(lambda x : x['name'] == "", args))

		#Filter out non tensors
		args = list(filter(lambda x : x['type'] == "tensor", args))

		if (len(args) == 0):
			self.shape = [(1,)]
			self.type = "float32" #FIX

		elif (len(args) == 1):
			in0 = args[0]
			_,t0,s0,dt0 = Pointwise.foo(in0)
			assert (t0 == "tensor")
			self.shape = [s0,]
			self.type = dt0

		elif (len(args) == 2):
			in0,in1 = args
			_,t0,s0,dt0 = Pointwise.foo(in0)
			_,t1,s1,dt1 = Pointwise.foo(in1)
			assert (t0 == t1 == "tensor")
			assert (dt0 == dt1)
			self.shape = [s0,s1]
			self.type = dt0

		elif (len(args) == 3):
			in0,in1,in2 = args
			_,t0,s0,dt0 = Pointwise.foo(in0)
			_,t1,s1,dt1 = Pointwise.foo(in1)
			_,t2,s2,dt2 = Pointwise.foo(in2)
			assert (t0 == t1 == t2 == "tensor")
			assert (dt0 == dt1 == dt2)
			self.shape = [s0,s1,s2]
			self.type = dt0
		else:
			assert False
		return

	def params(self):
		p = OrderedDict([('T',self.shape), ('type', self.type)])
		return p

	def tc(self):
		return "-"

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

	def elems(self):
		tensor = self.shape
		t = self.type

		if (len(tensor) == 1):
			elems = 2 * Utility.numElems(tensor[0])
		elif (len(tensor) == 2):
			if (tensor[0] == tensor[1]):	# same shape
				elems = Utility.numElems(tensor[0])
				if self.dir == "fprop":
					elems *= 3
				else:
					if (self.op_ in ["add", "__add__", "sub", "__sub__", "__isub__"]):
						elems *= 2
					elif (self.op_ in ["__mul__", "__rmul__", "div", "__truediv__"]):
						elems *= 3
					else:
						assert False
			else:	#check for broadcast conditions
				array1 = np.empty(list(tensor[0]))
				array2 = np.empty(list(tensor[1]))
				try:
					out = np.broadcast(array1, array2).shape
				except:
					assert False

				elems = Utility.numElems(tensor[0])
				elems += Utility.numElems(tensor[1])
				elems += Utility.numElems(out)
				#TODO bprop
		elif (len(tensor) == 3):
			if (tensor[0] == tensor[1] == tensor[2]):	#same shape
				elems = Utility.numElems(tensor[0])
				elems *= 4
			else:
				assert False
		else:
			assert False

		return elems

	def bytes(self):
		return self.elems() * Utility.typeToBytes(self.type)

	def flops(self):
		# Note: some cases may still be missing.

		f = 0
		if self.op_ in ["__abs__", "__neg__", "__add__", "__sub__", "__mul__",
					"__radd__", "__rmul__", "__iadd__", "__isub__", "__imul__", "__itruediv__",
					"abs", "abs_", "neg", "neg_", "add", "add_", "div", "div_", "mul", "mul_",
					"sub", "sub_", "exp", "exp_", "sign", "sign_", "trunc", "trunc_",
					"sin", "sin_", "cos", "cos_", "sinh", "sinh_", "cosh", "cosh_",
					"sqrt", "sqrt_", "rsqrt", "rsqrt_", "__lt__", "__gt__", "__ge__", "__le__",
					"__eq__", "__ne__", "lt", "lt_", "gt", "gt_", "ge", "ge_", "le", "le_",
					"eq", "eq_", "ne", "ne_", "ceil", "ceil_", "clamp", "clamp_", "floor", "floor_",
					"round", "sign", "sign_", "trunc", "trunc_"]:
			# We're counting only one operand, not two (2 operands, 1 op)
			f = self.elems() / 2
		elif self.op_ in ["fmod", "fmod_"]:
			f = self.elems()
		elif self.op_ in ["tanh", "tanh_", "sigmoid", "sigmoid_", "log", "log_", "log2",
			 "log2_", "log10", "log10_"]:
			f = self.elems() * 2
		elif self.op_ in ["asin", "asin_", "acos", "acos_", "atan", "atan_"]:
			# no intrinsic, hence slow execution
			# surprisingly, asin/acos and atan were all the same (via nvprof measurement)
			f = self.elems() * 10

		return f
