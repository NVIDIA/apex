from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Convert(OperatorLayerBase):
	"""
	Class to handle convert operations.
	"""
	ops = ["byte", "char", "double", "float", "half", "int", "long", "short", "to"]

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		assert (mod == "Tensor")
		assert (op in Convert.ops)
		assert (len(args) == 1)

		#The argument could be a tensor or scalar
		t = args[0]
		if t['type'] == "tensor":
			shape = t['shape']
			stype = t['dtype']
		else:
			shape = (1,)
			stype = t['type']
		if self.op_ == "to":
			op = stype

		self.shape = shape
		self.stype = stype
		self.dtype = op

	def params(self):
		p = OrderedDict([('T', self.shape), ('stype', self.stype), ('dtype', self.dtype)])
		return p

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

	def tc(self):
		return "-"

	def elems(self):
		return Utility.numElems(self.shape)

	def flops(self):
		return 0

	def bytes(self):
		b = self.elems() * (Utility.typeToBytes(self.stype) + Utility.typeToBytes(self.dtype))
		return b
