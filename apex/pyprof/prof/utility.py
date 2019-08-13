from functools import reduce

class Utility(object):

	@staticmethod
	def numElems(shape):
		assert (type(shape) == tuple)
		return reduce(lambda x,y: x*y, shape, 1)

	@staticmethod
	def typeToBytes(t):
		if (t in ["uint8", "int8", "byte", "char"]):
			return 1
		elif (t in ["float16", "half", "int16", "short"]):
			return 2 
		elif (t in ["float32", "float", "int32", "int"]):
			return 4
		elif (t in ["int64", "long", "float64", "double"]):
			return 8
		assert False

	@staticmethod
	def typeToString(t):
		if (t in ["uint8", "byte", "char"]):
			return "uint8"
		elif (t in ["int8",]):
			return "int8"
		elif (t in ["int16", "short",]):
			return "int16"
		elif (t in ["float16", "half"]):
			return "fp16"
		elif (t in ["float32", "float"]):
			return "fp32"
		elif (t in ["int32", "int",]):
			return "int32"
		elif (t in ["int64", "long"]):
			return "int64"
		elif (t in ["float64", "double",]):
			return "fp64"
		assert False

	@staticmethod
	def hasNVTX(marker):
		if type(marker) is str:
			try:
				marker = eval(marker)
			except:
				return False

		if type(marker) is dict:
			keys  = marker.keys()
			return ("mod" in keys) and ("op" in keys) and ("args" in keys)
		else:
			return False

	@staticmethod
	def isscalar(t):
		return (t in ["float", "int"])
