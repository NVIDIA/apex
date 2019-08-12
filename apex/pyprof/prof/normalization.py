from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class BatchNorm(OperatorLayerBase):

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		assert (op == "batch_norm")
		assert (len(args) == 8)
		i = args[0]
		assert (i['type'] == "tensor")

		self.shape = i['shape']
		self.type = i['dtype']
		self.dir = d.dir

	def params(self):
		p = OrderedDict([('T', self.shape), ('type', self.type)])
		return p

	def tc(self):
		return "-"

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

	def elems(self):
		return Utility.numElems(self.shape)

	def flops(self):
		# Variance algo-dependent, but this is a reasonable value.
		return self.elems() * 8

	def bytes(self):
		e = self.elems()
		if self.dir == "fprop":
			e *= 4
		else:
			e *= 5

		return e * Utility.typeToBytes(self.type)
