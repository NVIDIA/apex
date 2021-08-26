import apex.mpu as mpu
import torch
from apex.mpu.tests.commons import initialize_distributed

from apex.mpu import mappings
from apex.mpu.tests import global_vars
global_vars.set_global_variables()

def test__reduce(args, tensor_model_parallel_size):
	print("Testing reduction size =",tensor_model_parallel_size)
	mpu.initialize_model_parallel(tensor_model_parallel_size)
	tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
	assert torch.equal(mappings._reduce(torch.full((10,10,10,10),(50))), torch.full((10,10,10,10),50* tensor_model_parallel_size))
	mpu.destroy_model_parallel()
	print("Passed!")

def test__split(args, tensor_model_parallel_size):
	print("Testing splitting size =",tensor_model_parallel_size)
	mpu.initialize_model_parallel(tensor_model_parallel_size)
	tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
	listy = []
	for i in range(tensor_model_parallel_size):
		listy.append(torch.randn(10,1))
	x = torch.cat(tuple(listy), 1)
	out = mappings._split(x)
	assert torch.equal(out, listy[mpu.get_tensor_model_parallel_rank()])
	mpu.destroy_model_parallel()
	print("Passed!")

def test__gather(args, tensor_model_parallel_size):

	print("Testing gathering size =",tensor_model_parallel_size)
	mpu.initialize_model_parallel(tensor_model_parallel_size)
	tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()
	assert torch.equal(mappings._gather(torch.tensor([mpu.get_tensor_model_parallel_rank()])), torch.tensor(list(range(tensor_model_parallel_size))))
	mpu.destroy_model_parallel()
	print("Passed!")

if __name__ == '__main__':
	initialize_distributed()

	world_size = torch.distributed.get_world_size()
	args=global_vars.get_args()
	tensor_model_parallel_size = 1
	while tensor_model_parallel_size <= world_size:
		test__reduce(args, tensor_model_parallel_size)
		test__split(args, tensor_model_parallel_size)
		test__gather(args, tensor_model_parallel_size)
		tensor_model_parallel_size *= 2	
	print(">> passed the test :-)")
