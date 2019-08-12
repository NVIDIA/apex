from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

def hasTileSize(name):
	if ("sgemm" in name) or ("884gemm" in name) or ("hgemm" in name):
		return True
	else:
		return False

def ctaTile(name):
	name = name.split("_")
	name = list(filter(lambda x : "x" in x, name))
	name = list(filter(lambda x : "slice" not in x, name))
	assert(len(name) == 1)
	name = name[0].split("x")
	assert(len(name) == 2)
	name = list(map(int, name))
	return name[0], name[1]

class RNNCell(OperatorLayerBase):
	"""
	This class supports RNNCell, LSTMCell and GRUCell.
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

		self.name = d.name
		self.dir = d.dir
		self.sub = d.sub
		self.grid = d.grid

		assert (op == "forward")
		assert (mod in ["LSTMCell", "GRUCell", "RNNCell"])
		assert (len(args) in [2,3])

		x,h = args[0],args[1]
		b1,ii = x['shape']
		b2,hh = h['shape']
		assert b1 == b2
		assert x['dtype'] == h['dtype']
		t = x['dtype']

		self.cell = mod
		self.inp = ii
		self.hid = hh
		self.b = b1
		self.type = t

		self.multiple = 1
		if self.cell == "LSTMCell":
			self.multiple = 4
		elif self.cell == "GRUCell":
			self.multiple = 3

		self.gemm = None
		self.m = None
		self.n = None
		self.k = None
		self.elems = 0

		self.bar()
		
	def params(self):
		if self.gemm is None:
			p = OrderedDict([('cell', self.cell), ('X', self.inp), ('H', self.hid), ('B', self.b), ('type', self.type)])
		else:
			assert self.m is not None
			assert self.n is not None
			assert self.k is not None
			p = OrderedDict([('gemm', self.gemm), ('M', self.m), ('N', self.n), ('K', self.k), ('type', self.type)])
		return p

	def tc(self):
		if "gemm" in self.name:
			return 1 if "884gemm" in self.name else 0
		else:
			return "-"

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_

	def bytes(self):
		if self.gemm is not None:
			m, n, k, t = self.m, self.n, self.k, self.type
			b = (m*k + k*n + m*n) * Utility.typeToBytes(t)
		elif self.elems != 0:
			b = self.elems * Utility.typeToBytes(self.type)
		else:
			b = 0
		return b

	def flops(self):
		if self.gemm is not None:
			m, n, k = self.m, self.n, self.k
			f = 2*m*n*k
		elif self.elems != 0:
			f = 0 #TODO
		else:
			f = 0
		return f

	def bar(self):
		cell = self.cell
		X = self.inp
		H = self.hid
		B = self.b
		t = self.type
		subseqId = self.sub
		direc = self.dir
		name = self.name
		grid = self.grid
		multiple = self.multiple

		if direc == "fprop":
			subseqId = subseqId % 3
			if subseqId == 0: #layer gemm
				self.gemm = "layer"
				self.m = multiple*H
				self.n = B
				self.k = X
			elif subseqId == 1: #recurrent gemm
				self.gemm = "recur"
				self.m = multiple*H
				self.n = B
				self.k = H
			else:
				layerGemmElems = multiple*H*B
				recurGemmElems = multiple*H*B
				cElems = H*B
				hElems = H*B
				totElems = layerGemmElems + recurGemmElems + 2*cElems + hElems
				self.elems = totElems

		else:
			if ("gemm" in name) and hasTileSize(name):	#gemm
				#Get cta tile size
				tileX, tileY = ctaTile(name)
				#Get grid dimensions
				grid = grid.split(",")
				gridX,gridY,gridZ = map(lambda x : int(x), grid)

				gemmM = tileX * gridX
				gemmN = tileY * gridY

				if name[-3:] == "_nn": # dgrad
					if (gemmM == H):	# recurrent dgrad
						#Ideally gemmN = B, but we have a limited set of tile sizes.
						gemmN = B
						gemmK = multiple*H

						self.gemm = "recur"
						self.m = gemmM
						self.n = gemmN
						self.k = gemmK

					elif (gemmM == X):	# layer dgrad
						#assert(gemmN % B == 0)
						gemmK = multiple*H

						self.gemm = "layer"
						self.m = gemmM
						self.n = gemmN
						self.k = gemmK

					else:
						pass

				elif name[-3:] == "_nt": #wgrad
					if (gemmM == H):	#recurrent wgrad
						assert (gemmN == multiple*H)
						gemmK = B

						self.gemm = "recur"
						self.m = gemmM
						self.n = gemmN
						self.k = gemmK

					elif (gemmM == X):	#layer wgrad
						assert (gemmN == multiple*H)
						gemmK = B

						self.gemm = "layer"
						self.m = gemmM
						self.n = gemmN
						self.k = gemmK

					else:
						pass
				else:
					pass
			else:
				pass

		return
