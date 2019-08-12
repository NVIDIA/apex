from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Embedding(OperatorLayerBase):

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
		assert (op == "embedding")

		self.ishape = args[0]['shape']
		self.itype = args[0]['dtype']

		self.eshape = args[1]['shape']
		self.etype = args[1]['dtype']

		assert (len(self.eshape) == 2)

		self.dir = d.dir
		self.sub = d.sub
		return

	def params(self):
		p = OrderedDict([('I', self.ishape), ('itype', self.itype), ('E', self.eshape), ('etype', self.etype)])
		return p

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

	def tc(self):
		return "-"

	def bytes(self):
		ishape = self.ishape
		itype = self.itype
		eshape = self.eshape
		etype = self.etype

		ielems = Utility.numElems(ishape)

		b = 0
		if self.dir == "fprop":
			#indices
			b += ielems * Utility.typeToBytes(itype)
			#read and write the embedding matrix
			b += ielems * eshape[1] * 2 * Utility.typeToBytes(etype)
		else:
			#3 times the size of the incoming gradient
			b = ielems * eshape[1] * 3 * Utility.typeToBytes(etype)

			if self.sub > 0:
				b = 0

		return b

	def flops(self):
		# Note: not implemented yet
		return 0
