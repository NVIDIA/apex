from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Activation(OperatorLayerBase):
	"""
	This class handles the various activation functions.
	"""

	ops = ["celu", "elu", "elu_", "hardshrink", "hardtanh", "hardtanh_", "leaky_relu", "leaky_relu_", "logsigmoid", "prelu", "relu", "relu_", "relu6", "rrelu", "rrelu_", "selu", "sigmoid", "softplus", "softshrink", "softsign", "tanh", "tanhshrink", "threshold", "threshold_"]

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		assert (mod in ["torch.nn.functional", "torch", "Tensor"])

		#Filter out named parameters
		args = list(filter(lambda x : x['name'] == '', args))

		assert (len(args) >= 1)
		arg = args[0]
		assert (arg['type'] == "tensor")

		self.i = arg
		self.dir = d.dir

	def params(self):
		p = OrderedDict([('T', self.i['shape']),('type', self.i['dtype'])])
		return p

	def flops(self):
		direction = self.dir
		tensor = self.i['shape']
		t = self.i['dtype']

		# TODO: revise
		elems = Utility.numElems(tensor)
		return elems

	def bytes(self):
		direction = self.dir
		tensor = self.i['shape']
		t = self.i['dtype']

		elems = Utility.numElems(tensor)
		elems = elems * (2 if direction == "fprop" else 3)

		return elems * Utility.typeToBytes(t)

	def tc(self):
		return "-"

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_
