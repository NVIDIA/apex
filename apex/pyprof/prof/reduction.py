from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Mean(OperatorLayerBase):

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		assert (mod in ["torch", "Tensor"])
		assert (op == "mean")

		#Filter out named parameters
		args = list(filter(lambda x : x['name'] == '', args))

		assert (len(args) <= 2)
		i = args[0]

		self.shape = i['shape']
		self.type = i['dtype']
		self.dir = d.dir
		self.sub = d.sub

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

	def bytes(self):
		if self.sub == 0:
			return self.elems() * Utility.typeToBytes(self.type)
		else:
			return 0

	def flops(self):
		if self.sub == 0:
			return self.elems() + 1
		else:
			return 0

class Sum(OperatorLayerBase):

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		assert (mod in ["torch", "Tensor"])
		assert (op == "sum")
		assert (len(args) >= 1)

		#Get input
		if (args[0]['name'] == ""):
			i = args[0]
		else:
			i = list(filter(lambda x : x['name'] == "input", args))[0]

		self.shape = i['shape']
		self.type = i['dtype']

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
		# Note: This is incorrect, need to calculate actual flops (say via nvprof)
		return self.elems()

	def bytes(self):
		return self.elems() * Utility.typeToBytes(self.type)

class Norm(OperatorLayerBase):

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		assert (mod in ["torch", "Tensor"])
		assert (op == "norm")
		#assert (len(args) == 1)
		i = args[0]
		self.shape = i['shape']
		self.type = i['dtype']

	def params(self):
		p = OrderedDict([('T', self.shape), ('type', self.type)])
		return p

	def elems(self):
		return Utility.numElems(self.shape)

	def bytes(self):
		return self.elems() * Utility.typeToBytes(self.type)

	def flops(self):
		# square and add plus sqrt
		return 2 * self.elems() + 1

	def tc(self):
		return "-"

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_
