import inspect
import unittest

from apex.pyprof.prof.data import Data
from apex.pyprof.prof.prof import foo


class TestPyProfData(unittest.TestCase):

	def __init__(self, testName):
		super().__init__(testName)

	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_data(self):
		kernels = [
			{'kShortName': 'elementwise_kernel', 'kDuration': 2848, 'layer': [], 'trace': [], 'reprMarkers': [], 'marker': ["{'mod': 'Tensor', 'op': 'float', 'args': [{'name': '', 'type': 'tensor', 'shape': (18, 104, 160), 'dtype': 'bool'}]}"], 'seqMarker': ['to, seq = 60471'], 'seqId': [60471], 'subSeqId': 0, 'altSeqId': [], 'dir': 'fprop', 'mod': ['Tensor'], 'op': ['float'], 'tid': 1431533376, 'device': 0, 'stream': 7, 'grid': (585, 1, 1), 'block': (512, 1, 1), 'kLongName': 'void at::native::elementwise_kernel<512, 1, void at::native::gpu_kernel_impl<void at::native::copy_kernel_impl<float, bool>(at::TensorIterator&)::{lambda(bool)#1}>(at::TensorIterator&, void at::native::copy_kernel_impl<float, bool>(at::TensorIterator&)::{lambda(bool)#1} const&)::{lambda(int)#1}>(int, void at::native::gpu_kernel_impl<void at::native::copy_kernel_impl<float, bool>(at::TensorIterator&)::{lambda(bool)#1}>(at::TensorIterator&, void at::native::copy_kernel_impl<float, bool>(at::TensorIterator&)::{lambda(bool)#1} const&)::{lambda(int)#1})'},
			{'kShortName': 'elementwise_kernel', 'kDuration': 201182, 'layer': [], 'trace': [], 'reprMarkers': [], 'marker': ["{'mod': 'Tensor', 'op': 'clone', 'args': [{'name': '', 'type': 'tensor', 'shape': (18, 4, 416, 640), 'dtype': 'float32'}]}"], 'seqMarker': ['clone, seq = 60161'], 'seqId': [60161], 'subSeqId': 0, 'altSeqId': [], 'dir': 'fprop', 'mod': ['Tensor'], 'op': ['clone'], 'tid': 1431533376, 'device': 0, 'stream': 7, 'grid': (37440, 1, 1), 'block': (128, 1, 1), 'kLongName': 'void at::native::elementwise_kernel<128, 4, void at::native::gpu_kernel_impl<void at::native::copy_kernel_impl<float, float>(at::TensorIterator&)::{lambda(float)#1}>(at::TensorIterator&, void at::native::copy_kernel_impl<float, float>(at::TensorIterator&)::{lambda(float)#1} const&)::{lambda(int)#2}>(int, void at::native::gpu_kernel_impl<void at::native::copy_kernel_impl<float, float>(at::TensorIterator&)::{lambda(float)#1}>(at::TensorIterator&, void at::native::copy_kernel_impl<float, float>(at::TensorIterator&)::{lambda(float)#1} const&)::{lambda(int)#2})'},
		]

		for k in kernels:
			d = Data(k)
			mod = k['mod']
			op = k['op']
			xx = foo(mod, op, d)
			d.setParams(xx.params())


def run_tests(test_name):
	dummy = TestPyProfData(test_name)
	test_cases = list(filter(lambda x: 'test_' in x, map(lambda x: x[0], inspect.getmembers(dummy, predicate=inspect.ismethod))))
	print(f'Running tests for {test_name}')
	suite = unittest.TestSuite()
	for test_case in test_cases:
		suite.addTest(TestPyProfData(test_case))
	unittest.TextTestRunner().run(suite)

if __name__ == '__main__':
	run_tests('test_data')
