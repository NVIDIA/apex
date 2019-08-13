from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Dropout(OperatorLayerBase):

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		assert (mod == "torch.nn.functional")
		assert (op == "dropout")
		#assert (len(args) == 1)

		self.shape = args[0]['shape']
		self.type  = args[0]['dtype']
		self.dir = d.dir

		return

	def params(self):
		p = OrderedDict([('T', self.shape), ('type', self.type)])
		return p

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

	def tc(self):
		return "-"

	def elems(self):
		return Utility.numElems(self.shape)

	def bytes(self):
		#Ignoring the cost of writing and reading the mask
		return Utility.typeToBytes(self.type) * self.elems() * 2

	def flops(self):
		# Note: This is approximate and depends on the RNG
		return 5*self.elems()
