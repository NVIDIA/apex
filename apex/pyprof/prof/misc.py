from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Foo(OperatorLayerBase):
	"""
	An object of Foo is instantiated when we detect an unsupported operator.
	"""
	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		shapes = []
		types = []

		for arg in args:
			if arg['type'] == "tensor":
				shapes.append(arg['shape'])
				types.append(arg['dtype'])

		self.shape = shapes
		self.type = types

	def params(self):
		p = OrderedDict([('T', self.shape), ('type', self.type)])
		return p

	def tc(self):
		return "-"

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

	def flops(self):
		return 0

	def bytes(self):
		return 0

class Copy(OperatorLayerBase):

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
		assert (op == "copy_")
		assert (len(args) == 2)

		dst, src = args
		assert (src['type'] == dst['type'])
		assert (src['shape'] == dst['shape'])

		self.shape = src['shape']
		self.stype = src['dtype']
		self.dtype = dst['dtype']

	def params(self):
		#The data type might be different
		p = OrderedDict([('T', self.shape), ('stype', self.stype), ('dtype', self.dtype)])
		return p

	def tc(self):
		return "-"

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

	def flops(self):
		return 0

	def elems(self):
		return Utility.numElems(self.shape)

	def bytes(self):
		return self.elems() * (Utility.typeToBytes(self.stype) + Utility.typeToBytes(self.dtype))

class Clone(OperatorLayerBase):

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
		assert (op == "clone")
		assert (len(args) == 1)
		t = args[0]
		self.shape = t['shape']
		self.type = t['dtype']

	def params(self):
		p = OrderedDict([('T', self.shape), ('type', self.type)])
		return p

	def flops(self):
		return 0

	def tc(self):
		return "-"

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

	def elems(self):
		return Utility.numElems(self.shape)

	def bytes(self):
		return 2 * self.elems() * Utility.typeToBytes(self.type)

class Contiguous(OperatorLayerBase):

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
		assert (op == "contiguous")
		assert (len(args) == 1)
		t = args[0]
		self.shape = t['shape']
		self.type = t['dtype']

	def params(self):
		p = OrderedDict([('T', self.shape), ('type', self.type)])
		return p

	def flops(self):
		return 0

	def bytes(self):
		return 2 * Utility.numElems(self.shape) * Utility.typeToBytes(self.type)

	def tc(self):
		return "-"

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

class Any(OperatorLayerBase):

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
		assert (op == "any")
		assert (len(args) == 1)	#could be 2 as well, the second argument is a bool
		t = args[0]

		self.shape = t['shape']
		self.type = t['dtype']
		self.sub = d.sub
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

	def flops(self):
		return 0

	def bytes(self):
		return Utility.numElems(self.shape) * Utility.typeToBytes(self.type)
