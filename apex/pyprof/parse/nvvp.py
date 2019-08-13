import sys

class NVVP(object):
	"""
	This class gets kernel information from the SQL (nvvp) database.
	"""

	driverT = "CUPTI_ACTIVITY_KIND_DRIVER"
	runtimeT = "CUPTI_ACTIVITY_KIND_RUNTIME"
	kernelT = "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"
	markerT = "CUPTI_ACTIVITY_KIND_MARKER"
	stringT = "StringTable"

	def __init__(self, db):
		self.db = db
		self.markerId = 0

	def getProfileStart(self):
		"""
		Get the profile start time
		"""
		profStart = sys.maxsize
		for table in [self.driverT, self.runtimeT, self.kernelT, self.markerT]:
			colname = "timestamp" if table is self.markerT else "start"
			cmd = "select {} from {} ORDER BY {} ASC LIMIT 1".format(colname, table, colname)
			result = self.db.select(cmd)
			assert(len(result) <= 1)
			if (len(result) == 1):
				assert(colname in result[0])
				t = result[0][colname]
				if (t < profStart):
					profStart = t
		assert(profStart < sys.maxsize)
		return profStart

	def getString(self, id_):
		"""
		Get the string associated with an id.
		"""
		cmd = "select value from {} where _id_ = {}".format(self.stringT, id_)
		result = self.db.select(cmd)
		assert (len(result) == 1)
		return result[0]['value']

	def createMarkerTable(self):
		"""
		Create a temporary table and index it to speed up repeated SQL quesries.
		The table is an INNER JOIN of CUPTI_ACTIVITY_KIND_MARKER with itself.
		"""
		cmd = 'CREATE TEMPORARY TABLE marker AS SELECT \
					a._id_ as id, \
					a.timestamp AS startTime, \
					b.timestamp AS endTime, \
					HEX(a.objectId) AS objectId, \
					a.name AS name \
					FROM {} AS a INNER JOIN {} AS b ON \
					a.id = b.id and \
					a.flags = 2 and b.flags = 4'.format(self.markerT, self.markerT)
		self.db.execute(cmd)

		self.db.execute('CREATE INDEX start_index ON marker (startTime)')
		self.db.execute('CREATE INDEX end_index ON marker (endTime)')
		self.db.execute('CREATE INDEX id_index ON marker (id)')

	def getCPUInfo(self, corrId):
		"""
		Given the correlation id, get CPU start, end, thread id, process id.
		The information can be in the runtime table or the driver table.
		"""

		#First look in the runtime table
		cmd = "select start,end,processId,threadId from {} where correlationId={}".format(self.runtimeT, corrId);
		result = self.db.select(cmd)
		assert (len(result) <= 1)

		if (len(result) == 0):
			#Look in the driver table
			cmd = "select start,end,processId,threadId from {} where correlationId={}".format(self.driverT, corrId);
			result = self.db.select(cmd)

		assert (len(result) == 1)
		info = result[0]
		start = info['start']
		end = info['end']
		pid = info['processId']
		tid = info['threadId']
		tid = tid & 0xffffffff	#convert to unsigned
		assert (end > start)
		return [start, end, pid, tid]

	def getKernelInfo(self):
		"""
		Get GPU kernel info
		"""
		cmd = "select name,correlationId,start,end,deviceId,streamId,gridX,gridY,gridZ,blockX,blockY,blockZ from {}".format(self.kernelT)
		result = self.db.select(cmd)
		return result

	def getMarkerInfo(self, objId, startTime, endTime):
		"""
		This function first finds all NVTX markers encapsulating
		a runtime / driver kernel launch.
		It then splits the markers into many lists.
			layerMarkers : User added NVTX markers
			traceMarkers : Call trace markers (inserted by pyprof)
			reprMarkers  : Markers containing the extra_repr() of a module (inserted by pyprof)
			pyprofMarkers: Markers containing args and kwargs (tensor shape, datatype etc.)
			seqMarkers   : Markers containing PyTorch internal sequence markers (inserted by PyTorch)
			altSeqMarkers: Markers inserted by PyTorch between two kernel launches. Needs better explanation.
			otherMarkers : Markers not in either of the above categories.

		We extract seqId from the seq and altSeq markers. The seqId is used in bprop.
		We also extract information from the layerMarkers.
		"""

		layerMarkers = []
		traceMarkers = []
		reprMarkers = []
		pyprofMarkers = []
		seqMarkers = []
		otherMarkers = []
		altSeqMarkers = []
		bprop = False

		#Helper functions

		def delete(objId, sTime):
			"""
			Delete rows from the temporary SQL table which are no longer required.
			This speeds up future queries.
			"""
			margin = 0
			cmd = 'DELETE FROM marker WHERE objectId = "{}" AND endTime < {}'.format(objId, sTime - margin)
			#cmd = 'DELETE FROM marker WHERE endTime < {}'.format(sTime - margin)
			self.db.execute(cmd)

		def getLayerName(mlist):
			"""
			Get layer names from layer marker list.
			"""
			layers = []
			assert(type(mlist) == list)
			for m in mlist:
				assert("layer:" in m)
				l = m.split(":")[1]
				layers.append(l)
			return layers

		def getSeqId(mlist):
			"""
			Get sequence ids from seq / alt seq marker list.
			"""
			ids = []
			assert(type(mlist) == list)
			for m in mlist:
				assert(", seq = " in m)
				seq = int(m.split("=")[1])
				ids.append(seq)

			#Remove duplicates
			ids = list(set(ids))
			ids.sort()
			return ids

		def seqcompare(elem):
			"""
			Sorting function for sequence markers
			"""
			assert (", seq = " in elem)
			#sort by sequence id and then the string
			l = elem.split(" = ")
			return l[1] + l[0]

		def prune(mlist):
			"""
			Remove markers with the same seqId and if the strings are similar.
			This function works on a sorted sequence.
			"""
			assert (type(mlist) == list)
			assert (len(mlist))
			a = mlist[0:1]
			for i in range(1,len(mlist)):
				m = mlist[i]
				pm = mlist[i-1]
				name,seq = m.split(",")
				pname,pseq = pm.split(",")
				similar = (name in pname) or (pname in name)
				if (seq == pseq) and similar:
					continue
				else:
					a.append(m)
			return a

		def filterTrace(mlist):
			"""
			Filter trace markers to remove certain file names.
			"""
			assert (type(mlist) == list)
			if len(mlist) == 0:
				return mlist
			mlist = mlist[-1]	#The last stack trace will be a super set.
			mlist = eval(mlist)
			mlist = mlist['traceMarker']
			assert (type(mlist) == list)
			mlist = list(filter(lambda x : "/torch/nn/modules/" not in x, mlist))
			mlist = list(filter(lambda x : "/torch/nn/functional.py" not in x, mlist))
			mlist = list(filter(lambda x : "/torch/tensor.py" not in x, mlist))
			mlist = list(filter(lambda x : "/torch/autograd/__init__.py" not in x, mlist))
			mlist = list(filter(lambda x : "/torch/_jit_internal.py" not in x, mlist))
			mlist = list(filter(lambda x : "/pyprof/nvtx/nvmarker.py" not in x, mlist))
			mlist = list(filter(lambda x : "/apex/optimizers/" not in x, mlist))
			mlist = list(filter(lambda x : "/torch/_utils.py" not in x, mlist))
			mlist = list(filter(lambda x : "/torch/optim/" not in x, mlist))
			return mlist

		#Find all encapsulating markers
		cmd = 'SELECT id,name from marker where \
				objectId = "{}" and \
				startTime < {} and \
				endTime > {} \
				ORDER BY startTime ASC'.format(objId, startTime, endTime)
		result = self.db.select(cmd)

		#Bin markers into different lists
		for r in result:
			m = self.getString(r['name'])

			#Hack: If its a known gradient checkpointing marker, ignore it.
			if m.find("CheckpointFunctionBackward") >= 0:
				continue

			if ("_backward, seq =" in m) or ("Backward, seq =" in m) or ("Backward0, seq =" in m):
				bprop = True

			if ("mod" in m) and ("op" in m) and ("args" in m) and ("type" in m):
				pyprofMarkers.append(m)
			elif ("layer:" in m):
				layerMarkers.append(m)
			elif ("traceMarker" in m):
				traceMarkers.append(m)
			elif ("strRepr" in m):
				reprMarkers.append(m)
			elif (", seq = " in m):
				seqMarkers.append(m)
			else:
				otherMarkers.append(m)

		#Remove duplicates, sort and prune seqMarkers
		if (len(seqMarkers)):
			seqMarkers = list(set(seqMarkers))
			seqMarkers.sort(key=seqcompare)
			seqMarkers = prune(seqMarkers)

		#Remove duplicates from otherMarkers
		otherMarkers = list(set(otherMarkers))

		#Get markers with seq id (inserted by PyTorch) from the previous kernel to the present kernel
		#Only for fprop kernels
		if (len(result) and not bprop):
			loId = self.markerId
			hiId = result[-1]['id']
			self.markerId = hiId
			
			#Get markers between loId and hiId
			cmd = 'SELECT id,name from marker where objectId = "{}" and id > {} and id < {} ORDER BY startTime ASC'.format(objId, loId, hiId)
			result1 = self.db.select(cmd)

			for r in result1:
				m = self.getString(r['name'])
				#Get only markers with seq id
				if (", seq=" in m):
					altSeqMarkers.append(m)

			#Remove duplicates, sort and prune altSeqMarkers
			if (len(altSeqMarkers)):
				altSeqMarkers = list(set(altSeqMarkers))
				altSeqMarkers.sort(key=seqcompare)
				altSeqMarkers = prune(altSeqMarkers)

		delete(objId, startTime)

		return layerMarkers, filterTrace(traceMarkers), reprMarkers, pyprofMarkers, seqMarkers, otherMarkers, altSeqMarkers, getSeqId(seqMarkers), getSeqId(altSeqMarkers), getLayerName(layerMarkers)
