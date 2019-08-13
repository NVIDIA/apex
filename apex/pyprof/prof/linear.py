from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Linear(OperatorLayerBase):

	'''
	Notes:
	If the bias occurs before the GEMM, then its 1 write (bias expansion).
	If the bias occurs after, then its 1 read and 1 write.
	bias in bprop is a reduction and hence is 1 read.
	'''

	gemmKernels = ["gemm", "gemv", "dot_kernel", "splitKreduce_kernel", "reduce_1Block_kernel"]
	biasKernels = ["kernelReduceContigDim", "kernelReduceNoncontigDim_shared", "elementwise_kernel", "reduce_kernel"]

	def setXWBMNK(self, args):
		x = None
		w = None
		b = None
		if (len(args) == 2):
			x,w = args
		elif (len(args) == 3):
			x,w,b = args
			assert (x['type'] == w['type'] == "tensor")
			if (b['type'] == "tensor"):
				assert(len(b['shape']) == 1)
			elif (b['type'] == "NoneType"):
				assert b['value'] is None
				b = None
			else:
				assert False
		else:
			assert False

		assert(len(w['shape']) == 2)
		k1 = x['shape'][-1]
		n,k2 = w['shape']
		assert(k1 == k2)
		if b is not None:
			assert(b['shape'][0] == n)
		t1 = x['dtype']
		t2 = w['dtype']
		assert(t1 == t2)

		# X, W, B
		self.x = x['shape']
		self.w = w['shape']
		self.b = b['shape'] if b is not None else None
		self.type = t1

		# M, N, K
		#n = Utility.numElems(x[0:-1])
		n = self.x[0:-1]
		k = self.x[-1]
		m,k1 = self.w
		assert (k == k1)

		self.m = m
		self.n = n
		self.k = k

	def tc(self):
		if self.op() == "linear":
			return 1 if "884gemm" in self.name else 0
		else:
			return "-"

	def __init__(self, d):
		self.name = d.name
		self.dir = d.dir
		self.sub = d.sub

		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		assert (mod == "torch.nn.functional")
		assert (op == "linear")

		self.setXWBMNK(args)

		if any(x in d.name for x in Linear.gemmKernels):
			self.op_ = "linear"
		else:
			assert (d.name in Linear.biasKernels)
			self.op_ = "bias"

		'''
		elif (("kernelPointwiseApply2" in d.name) or ("kernelReduceContigDim" in d.name) or ("kernelReduceNoncontigDim_shared" in d.name)):
			#bias expansion was before the gemm
			self.op_ = "bias"

		elif ("elementwise_kernel" in d.name):
			#Bias addition happens later with a broadcast tensor
			self.op_ = "bias"
			assert (len(d.argMarker) == 2)
			marker = eval(d.argMarker[1])
			mod = marker['mod']
			op = marker['op']
			args = marker['args']

			assert (mod == "Tensor")
			assert (op == "__iadd__")
			assert (len(args) == 2)
			mn = args[0]['shape']
			b = args[1]['shape']
			assert (len(b) == 1)

			assert (mn == (self.n + (self.m,)))
			assert (b == self.b)

		else:
			assert False
		'''

	def params(self):
		#p = OrderedDict([('X', self.x), ('W', self.w), ('B', self.b), ('type', self.type)])

		m, n, k, x, w, t = self.m, self.n, self.k, self.x, self.w, self.type
		if len(n) == 1:
			n = n[0]

		if self.op_ == "linear":
			if self.dir == "fprop":
				p = OrderedDict([('M', m), ('N', n), ('K', k), ('type', t)])
			elif self.dir == "bprop":
				if self.sub == 0:		#dgrad (most likely)
					p = OrderedDict([('M', k), ('N', n), ('K', m), ('type', t)])
				elif self.sub == 1:	#wgrad (most likely)
					p = OrderedDict([('M', k), ('N', m), ('K', n), ('type', t)])
				else:
					#This happens when there are additional kernels for reduction
					p = OrderedDict([('X', x), ('W', w), ('type', t)])
			else:
				assert False

		elif self.op_ == "bias":
			p = OrderedDict([('M', m), ('N', n), ('type', t)])
		else:
			assert False
		return p

	def op(self):
		return self.op_

	def bytesFlops(self):

		m = self.m
		n = Utility.numElems(self.n)
		k = self.k

		if self.op_ == "linear":
			if self.dir == "fprop":
				f = m * n * k * 2
				b = m*n + m*k + n*k * Utility.typeToBytes(self.type)
			elif self.dir == "bprop":
				if self.sub == 0:		#dgrad (most likely)
					f = m * n * k * 2
					b = m*n + m*k + n*k * Utility.typeToBytes(self.type)
				elif self.sub == 1:	#wgrad (most likely)
					f = m * n * k * 2
					b = m*n + m*k + n*k * Utility.typeToBytes(self.type)
				else:
					#This happens when there are additional kernels for reduction
					f = 0
					b = 0
			else:
				assert False

		elif self.op_ == "bias":
			f = m * n
			b = 2 * m * n * Utility.typeToBytes(self.type)
		else:
			assert False
		return b,f

	def bytes(self):
		b, f = self.bytesFlops()
		return b

	def flops(self):
		b, f = self.bytesFlops()
		return f

	def mod(self):
		return self.mod_
