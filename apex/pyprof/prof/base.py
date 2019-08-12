from abc import ABC, abstractmethod

class OperatorLayerBase(ABC):
	"""
	Base class for all layers and operators.
	Every derived class should have the following functions.
	"""

	@abstractmethod
	def tc(self):
		"""
		Tensor core usage by the kernel.
		Return "1" (yes), "0" (no, but possible), "-" (not applicable)
		"""
		pass

	@abstractmethod
	def params(self):
		"""
		Kernel parameters to be printed.
		"""
		pass

	@abstractmethod
	def flops(self):
		"""
		Note that 1 FMA = 2 flops.
		"""
		pass

	@abstractmethod
	def bytes(self):
		pass

	@abstractmethod
	def mod(self):
		"""
		Name of the module/class e.g. torch.nn.functional.
		"""
		pass

	@abstractmethod
	def op(self):
		"""
		Name of the operator e.g. sigmoid.
		"""
		pass
