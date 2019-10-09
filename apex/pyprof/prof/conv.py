from collections import OrderedDict
from .utility import Utility
from .base import OperatorLayerBase

class Conv(OperatorLayerBase):

	"""
	# N = batch size
	# C,H,W = input channels, height, width
	# K,P,Q = output channels, height, width
	# R,S = filter height, width
	# g = groups
	"""

	#todo: refine winograd and FFT
	convAuxList = ["nchwToNhwc", "nhwcToNchw", "OffsetsKernel",]
	winoAuxList = ["generateWinogradTilesKernel", "winogradWgradData", "winogradWgradOutput", "winogradWgradDelta"]
	fftAuxList = ["compute_gemm_pointers", "flip_filter", "fft2d_r2c_", "fft2d_c2r_", "fft1d_r2c", "fft1d_c2r"]
	miscAuxList = ["scaleTensor_kernel",]

	convList = ["_s884cudnn_", "_s1688cudnn_", "_scudnn_", "2d_grouped_direct_kernel", "cudnn::detail::implicit_convolve_sgemm", "cudnn::detail::dgrad2d_alg1_1", "cudnn::detail::wgrad_alg0_engine", "cudnn::detail::dgrad_engine", "dgrad_1x1_stride_2x2", "spatialDepthwiseConvolutionUpdateOutput"]
	winoList = ["winograd3x3Kernel", "_sgemm_"]
	fftList = ["fermiPlusCgemmLDS128_batched", "_gcgemm_",]
	miscList = []

	def __init__(self, d):
		marker = eval(d.argMarker[0])
		mod = marker['mod']
		op = marker['op']
		args = marker['args']

		self.marker = marker
		self.mod_ = mod
		self.op_ = op
		self.args = args

		self.dir = d.dir
		self.name = d.name
		self.sub = d.sub

		assert (mod == "torch.nn.functional")
		assert (op in ["conv1d", "conv2d"])
		length = len(args)
		assert (length >= 2) and (length <= 7)
		i,w = args[0], args[1]
		assert (i['type'] == "tensor")
		assert (w['type'] == "tensor")

		#ignore bias

		if (length >= 4) and (args[3]['name'] == ""):
			s = args[3]
		elif any(x['name'] == 'stride' for x in args):
			s = list(filter(lambda x : x['name'] == 'stride', args))[0]
		else:
			s = {'name': 'stride', 'type': 'int', 'value': 1}

		if (length >= 5) and (args[4]['name'] == ""):
			p = args[4]
		elif any(x['name'] == 'padding' for x in args):
			p = list(filter(lambda x : x['name'] == 'padding', args))[0]
		else:
			p = {'name': 'padding', 'type': 'int', 'value': 0}

		if (length >= 6) and (args[5]['name'] == ""):
			d = args[5]
		elif any(x['name'] == 'dilation' for x in args):
			d = list(filter(lambda x : x['name'] == 'dilation', args))[0]
		else:
			d = {'name': 'dilation', 'type': 'int', 'value': 1}

		if (length == 7) and (args[6]['name'] == ""):
			g = args[6]
		elif any(x['name'] == 'groups' for x in args):
			g = list(filter(lambda x : x['name'] == 'groups', args))[0]
		else:
			g = {'name': 'groups', 'type': 'int', 'value': 1}

		if op == "conv1d":
			assert (len(i['shape']) == 3)
			assert (len(w['shape']) == 3)
			assert (i['dtype'] == w['dtype'])
			N, C1, W = i['shape']
			K, C2, S = w['shape']
			assert (C1 == C2)
			p = p['value'] if Utility.isscalar(p['type']) else p['value'][0]
			s = s['value'] if Utility.isscalar(s['type']) else s['value'][0]
			d = d['value'] if Utility.isscalar(d['type']) else d['value'][0]
			g = g['value']
			assert (g == 1)
			H = 1
			R = 1

			P = 1 + (H - (((R-1))+1))
			Q = 1 + (W + 2*p - (((S-1)*d)+1))/s
			P = int(P)
			Q = int(Q)
			if (H == 1):
				assert (P == 1)
			if (W == 1):
				assert (Q == 1)

			self.N = N
			self.C = C1
			self.H = H
			self.W = W
			self.K = K
			self.P = P
			self.Q = Q
			self.R = R
			self.S = S
			self.ph = 0
			self.pw = p
			self.U = 1
			self.V = s
			self.dh = 1
			self.dw = d
			self.g = g
			self.type = i['dtype']

		elif op == "conv2d":
			assert (len(i['shape']) == 4)
			assert (len(w['shape']) == 4)
			assert (i['dtype'] == w['dtype'])
			N, C1, H, W = i['shape']
			K, C2, R, S = w['shape']

			if Utility.isscalar(p['type']):
				ph = pw = p['value']
			else:
				assert (p['type'] == "tuple")
				ph, pw = p['value']

			if Utility.isscalar(s['type']):
				sh = sw = s['value']
			else:
				assert (s['type'] == "tuple")
				sh, sw = s['value']

			if Utility.isscalar(d['type']):
				dh = dw = d['value']
			else:
				assert (d['type'] == "tuple")
				dh, dw = d['value']

			g = g['value']
			assert (g >= 1)
			assert (C1 == C2*g)

			P = 1 + (H + 2*ph - (((R-1)*dh)+1))/sh
			Q = 1 + (W + 2*pw - (((S-1)*dw)+1))/sw
			P = int(P)
			Q = int(Q)
			if (H == 1):
				assert (P == 1)
			if (W == 1):
				assert (Q == 1)

			self.N = N
			self.C = C1
			self.H = H
			self.W = W
			self.K = K
			self.P = P
			self.Q = Q
			self.R = R
			self.S = S
			self.ph = ph
			self.pw = pw
			self.U = sh
			self.V = sw
			self.dh = dh
			self.dw = dw
			self.g = g
			self.type = i['dtype']

		else:
			assert False

	def params(self):
		p = OrderedDict([('N',self.N), ('C',self.C), ('H',self.H), ('W',self.W), ('K',self.K), ('P',self.P), ('Q',self.Q), ('R',self.R), ('S',self.S), ('ph',self.ph), ('pw',self.pw), ('U',self.U), ('V',self.V), ('dh',self.dh), ('dw',self.dw), ('g',self.g), ('type',self.type)])
		return p

	def conv_bytes_flops(self, N, C, H, W, K, P, Q, R, S, g, t):
		f = 2*N*K*P*Q*C*R*S/g #for fprop
		elems = N*C*H*W + K*C*R*S/g + N*K*P*Q
		b = elems * Utility.typeToBytes(t)
		return b,f

	def bytes_flops(self):
		N,C,H,W,K,P,Q,R,S,ph,pw,U,V,dh,dw,g,t = self.params().values()

		if any(x in self.name for x in Conv.convAuxList+Conv.winoAuxList+Conv.fftAuxList+Conv.miscAuxList):
			bytes, flops = [0, 0]

		elif any(x in self.name for x in Conv.convList+Conv.winoList+Conv.fftList+Conv.miscList):
			if g == 1:
				bytes, flops = self.conv_bytes_flops(N,C,H,W,K,P,Q,R,S,g,t)
			else:
				if "2d_grouped_direct_kernel" in self.name:	#only 1 kernel is called
					bytes, flops = self.conv_bytes_flops(N,C,H,W,K,P,Q,R,S,g,t)
				elif "spatialDepthwiseConvolutionUpdateOutput" in self.name: #one kernel for separable conv
					bytes, flops = self.conv_bytes_flops(N,C,H,W,K,P,Q,R,S,g,t)
				else:	#a kernel per group is called
					bytes, flops = self.conv_bytes_flops(N,C/g,H,W,K/g,P,Q,R,S,1,t)

		elif ("calc_bias_diff" in self.name):	#bias gradient
			elems = N*K*P*Q
			flops = elems
			bytes = 2 * elems * Utility.typeToBytes(t)
			#params = OrderedDict([('N',N), ('K',K), ('P',P), ('Q',Q), ('type', t)])

		else:
			bytes, flops = [0, 0]

		return bytes, flops

	def bytes(self):
		b,_ = self.bytes_flops()
		return b

	def flops(self):
		_,f = self.bytes_flops()
		return f

	def tc(self):
		for s in ["884cudnn", "1688cudnn"]:
			if s in self.name:
				return 1
		return "-"

	def op(self):
		return self.op_

	def mod(self):
		return self.mod_
