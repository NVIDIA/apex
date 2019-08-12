from .collections import OrderedDict
from .utility import Utility

# Work in progress.

#poolFuncs = ["max_pool2d_with_indices_forward", "max_pool2d_with_indices"]
class MaxPool2d(object):

	def parse(marker):

		def convert2Tuple(arg):
			assert (arg['type'] in ["int", "tuple"])
			if arg['type'] == "int":
				return (arg['value'], arg['value'])
			else:
				return arg['value']

		mod = marker['mod']
		op = marker['op']
		args = marker['args']
		assert (mod == "torch.nn.functional")
		assert (op == "max_pool2d")
		assert (len(args) >= 2)

		#input
		assert (args[0]['name'] == "")
		inp = args[0]
		assert (inp['type'] == "tensor")
		i = inp['shape']
		t = inp['dtype']
		assert (len(i) == 4) #nchw tensor

		#kernel
		if (args[1]['name'] == ""):
			k = args[1]
		else:
			k = list(filter(lambda x : x['name'] == "kernel_size", args))[0]
		k = convert2Tuple(k)

		#stride
		s = k #default value
		if ((len(args) >= 3) and args[2] == ""):
			s = args[2]
			s = convert2Tuple(s)
		elif any(x['name'] == "stride" for x in args):
			s = list(filter(lambda x : x['name'] == "stride", args))[0]
			s = convert2Tuple(s)

		#padding
		p = (0,0)
		if ((len(args) >= 4) and args[3] == ""):
			p = args[3]
			p = convert2Tuple(p)
		elif any(x['name'] == "padding" for x in args):
			p = list(filter(lambda x : x['name'] == "padding", args))[0]
			p = convert2Tuple(p)
		
		params = OrderedDict([('T', i), ('K', k), ('s',s), ('p',p), ('type', t)])
		return params
