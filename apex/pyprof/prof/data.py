from .utility import Utility

class Data(object):
	"""
	Class to store all the data for every kernel e.g. name, bytes, flops, device, stream etc.
	"""
	def __init__(self, kernel):
		#Available from NVprof
		self.tid = kernel['tid']
		self.device = kernel['device']
		self.stream = kernel['stream']
		self.grid = str(kernel['grid']).replace(" ","").replace("(","").replace(")","")
		self.block = str(kernel['block']).replace(" ","").replace("(","").replace(")","")
		self.name = kernel['kShortName'].replace(" ","_")
		self.lName = kernel['kLongName']
		self.sil = kernel['kDuration']	#units ns

		self.index = None

		#Markers
		self.argMarker = kernel['marker']
		self.modMarker = kernel['reprMarkers']
		self.seqMarker = kernel['seqMarker']

		self.layer = kernel['layer']
		self.trace = kernel['trace']

		self.seqId = kernel['seqId']
		self.altSeqId = kernel['altSeqId']

		self.dir = kernel['dir']
		self.sub = kernel['subSeqId']

		self.mod = "na"
		self.op = "na"
		self.params = {"na":"na"}
		self.tc = "na"
		self.flops = 0
		self.bytes = 0

	def setParams(self, params):
		#Remove space from params
		qaz = ""
		for key,value in params.items():
			if "type" not in key:
				qaz += "{}={},".format(key,value)
			else:
				if type(value) is str:
					qaz += "{},".format(Utility.typeToString(value))
				else:
					qaz += "{}".format(value)

		self.params = qaz.replace(" ", "")

