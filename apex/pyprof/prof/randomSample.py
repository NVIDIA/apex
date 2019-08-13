from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class RandPerm(OperatorLayerBase):

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		assert (mod == "torch")
		assert (op == "randperm")
		assert (len(args) == 1)
		n = args[0]
		assert n['type'] == "int"
		self.n = n['value']

	def params(self):
		p = OrderedDict([('N', self.n)])
		return p

	def tc(self):
		return "-"

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

	def bytes(self):
		return self.n * Utility.typeToBytes("int64")

	def flops(self):
		# Depends on RNG but this is probably a reasonable assumption.
		return self.n * 3
