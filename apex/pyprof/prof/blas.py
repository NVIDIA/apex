from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase
import numpy as np

TC_GEMMS = ["884gemm", "1688gemm"]

class Addmm(OperatorLayerBase):

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		assert (mod in ["torch", "Tensor",])
		assert (op in ["addmm", "addmm_",])

		#Get alpha and beta
		alpha = 1
		beta = 1
		if any(x['name'] == 'alpha' for x in args):
			alpha = list(filter(lambda x : x['name'] == "alpha", args))[0]
			alpha = alpha['value']

		if any(x['name'] == 'beta' for x in args):
			beta = list(filter(lambda x : x['name'] == "beta", args))[0]
			beta = beta['value']

		self.alpha = alpha
		self.beta = beta

		#Filter out named parameters
		args = list(filter(lambda x : x['name'] == '', args))

		assert (len(args) == 3)
		C,A,B = args
		m,k1 = A['shape']
		k2,n = B['shape']
		assert (k1 == k2)
		t1 = A['dtype']
		t2 = B['dtype']
		t3 = C['dtype']
		assert(t1 == t2 == t3)

		self.A = A
		self.B = B
		self.C = C

		self.m = m
		self.n = n
		self.k = k1
		self.type = t1
		self.name = d.name

		return

	def tc(self):
            for s in TC_GEMMS:
                if s in self.name:
                    return 1
            return 0

	def bytes(self):
		m, n, k = self.m, self.n, self.k
		return Utility.typeToBytes(self.type) * (m*n + m*k + n*k)

	def flops(self):
		return self.m * self.n * self.k * 2

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

	def params(self):
		p = OrderedDict([('M',self.n),('N',self.m),('K',self.k),('type',self.type)])
		return p

class Bmm(OperatorLayerBase):

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		assert (mod == "torch") and (op == "bmm")

		#Filter out named params (kwargs)
		args = list(filter(lambda x : x['name'] == "", args))

		assert (len(args) == 2)
		A,B = args
		b1,m,k1 = A['shape']
		b2,k2,n = B['shape']
		assert (b1 == b2)
		assert (k1 == k2)
		t1 = A['dtype']
		t2 = B['dtype']
		assert(t1 == t2)

		self.A = A
		self.B = B
		self.b = b1
		self.m = m
		self.n = n
		self.k = k1
		self.type = t1
		self.name = d.name

	def tc(self):
            for s in TC_GEMMS:
                if s in self.name:
                    return 1
            return 0

	def params(self):
		#p = OrderedDict([('A', A['shape']), ('B', B['shape']), ('type', t1)])
		p = OrderedDict([('B',self.b), ('M',self.n),('N',self.m),('K',self.k),('type',self.type)])
		return p

	def flops(self):
		return self.b * self.m * self.n * self.k * 2

	def bytes(self):
		b, m, n, k = self.b, self.m, self.n, self.k
		return Utility.typeToBytes(self.type) * b * (m*n + m*k + n*k)

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

class Matmul(OperatorLayerBase):

	NON_GEMM = ["kernelPointwiseApply2", "reduce_1Block_kernel", "elementwise_kernel"]
	NON_TC = NON_GEMM + ["dot_kernel"]

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		self.name = d.name
		self.sub = d.sub

		assert ((mod == "torch") and (op == "matmul")) or ((mod == "Tensor") and (op == "__matmul__"))
		assert (len(args) == 2)

		assert any([x in d.name for x in Matmul.NON_TC + ["gemm", "gemv"]])

		A,B = args
		t1 = A['dtype']
		t2 = B['dtype']
		assert(t1 == t2)

		A = A['shape']
		B = B['shape']

		self.A = A
		self.B = B
		self.type = t1

		# batch, MNK
		if (len(A) == 1) and (len(B) == 1):
			#dot product
			assert (A[0] == B[0])
			self.b = (1,)
			self.m = 1
			self.n = 1
			self.k = A[0]

		elif (len(A) == 2) and (len(B) == 2):
			#gemm
			m,k1 = A
			k2,n = B
			assert(k1 == k2)
			self.b = (1,)
			self.m = m
			self.n = n
			self.k = k1

		elif (len(A) == 1) and (len(B) == 2):
			#vector matrix
			k1 = A[0]
			k2,n = B
			assert(k1 == k2)

			self.b = (1,)
			self.m = 1
			self.n = n
			self.k = k1

		elif (len(A) == 2) and (len(B) == 1):
			#gemv
			m,k1 = A
			k2 = B[0]
			assert (k1 == k2)

			self.b = (1,)
			self.m = m
			self.n = 1
			self.k = k1

		elif (len(A) == 1) and (len(B) > 2):
			assert (A[0] == B[-2])

			self.b = B[0:-2]
			self.m = 1
			self.n = B[-1]
			self.k = B[-2]

		elif (len(B) == 1) and (len(A) > 2):
			assert (B[0] == A[-1])

			self.b = A[0:-2]
			self.m = A[-2]
			self.n = 1
			self.k = A[-1]

		else:
			assert (len(A) >= 2)
			assert (len(B) >= 2)
			assert (A[-1] == B[-2])
			self.m = A[-2]
			self.n = B[-1]
			self.k = A[-1]

			aa = np.empty(A[0:-2])
			bb = np.empty(B[0:-2])
			self.b = np.broadcast(aa, bb).shape

	def params(self):
		return OrderedDict([('A', self.A), ('B', self.B), ('type', self.type)])

	def tc(self):
		if self.name in Matmul.NON_TC:
			return "-"
		else:
                    for s in TC_GEMMS:
                        if s in self.name:
                            return 1
                    return 0

	def bytes(self):
		# TODO: check bytes for non-GEMM cases
		if self.name in Matmul.NON_GEMM:
			return 2 * Utility.typeToBytes(self.type) * Utility.numElems(self.A) #could be B as well
		else:
			m, n, k = self.m, self.n, self.k
			return Utility.typeToBytes(self.type) * (m*n + m*k + n*k)

	def flops(self):
		# TODO: calculate actual FLOPs. At least we're not saying it's GEMM FLOPs for now.
		if self.name in Matmul.NON_GEMM:
			return 0
		else:
			return Utility.numElems(self.b) * self.m * self.n * self.k * 2

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

class Mm(OperatorLayerBase):

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		assert (mod == "torch") and (op == "mm")
		assert (len(args) == 2)

		A,B = args
		m,k1 = A['shape']
		k2,n = B['shape']
		assert (k1 == k2)
		t1 = A['dtype']
		t2 = B['dtype']
		assert(t1 == t2)

		self.A = A
		self.B = B
		self.m = m
		self.n = n
		self.k = k1
		self.type = t1
		self.name = d.name

		return

	def params(self):
		p = OrderedDict([('M',self.n),('N',self.m),('K',self.k),('type',self.type)])
		return p

	def tc(self):
            for s in TC_GEMMS:
                if s in self.name:
                    return 1
            return 0

	def bytes(self):
		m, n, k = self.m, self.n, self.k
		return Utility.typeToBytes(self.type) * (m*n + m*k + n*k)

	def flops(self):
		return self.m * self.n * self.k * 2

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_
