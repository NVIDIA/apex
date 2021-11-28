import torch
import torch.nn as nn
import time
import os
import traceback
import sys
import copy
import random
import numpy as np
import subprocess
from apex.transformer.testing.commons import TEST_SUCCESS_MESSAGE


def set_seed():
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0)
	torch.cuda.manual_seed_all(0)


def run_gpt(cmd):
	args = list(cmd.split(' '))
	p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	outs, errs = p.communicate()
	outs = list(str(outs).encode('ascii', 'replace').split('\n'))
	print(outs)
	quit()
	success = False
	runtime = 0
	init_dict = {'num_params':0}
	for out in outs:
		if "Average Iteration Time:" in str(out):
			runtime = float(out[out.find(':')+1:])
		if "Initialized GPT-2 w/:" in str(out):
			init_dict = json.loads(out[out.find(':')+1:])
		if str(out) == str(TEST_SUCCESS_MESSAGE):
			success=True
	return runtime, float(int(init_dict['num_params']))/10.0**9, success, errs


def plot(runtimes):
	import matplotlib.pyplot as plt
	for distributed_setting in runtimes.keys():
		plt.scatter(runtimes[distributed_setting].keys(), runtimes[distributed_setting].values(), label=distributed_setting)
	plt.legend()
	plt.xlabel('Parameters (Billions)')
	plt.ylabel('Training Iteration time (s)')
	plt.title(str("GPT Scaling w/ Offloading"))
	plt.savefig('offload_gpt_scaling.png')
	plt.close()


def main():
	runtimes = {}
	for data_parr, tens_parr, pipe_parr in [(8,1,1), (4,2,1), (2,1,4), (1,2,4)]:
		dist_setting = 'ddp=' + str(data_parr) + ', tensor_parr=' + str(tens_parr) + ', pipe_parr=' + str(pipe_parr)
		runtimes[dist_setting] = {} 
		print("Beginning Testing for", dist_setting)
		for n in range(500,1000000,500):
			cmd = "python3 -m torch.distributed.launch --nproc_per_node=8 run_gpt_minimal_test.py"
			cmd += " --micro-batch-size 1 --num-layers " + str(n) + " --hidden-size 128 --num-attention-heads 16"
			cmd += ' --max-position-embeddings 128 --seq-length 128 --tensor-model-parallel-size ' + str(tens_parr)
			cmd += " --pipeline-model-parallel-size " + str(pipe_parr)
			print(cmd)
			runtime, bill_params, success, errs = run_gpt(cmd)
			if success:
				runtimes[dist_setting][bill_params] = runtime
				print(runtime, 'ms')
			else:
				print("GPT-2 w/", n, "layers failed using", dist_setting)
				if sys.argv[1] == '-v' or sys.argv[1] == '--verbose':
					print("STDERR:")
					print(errs)
				print("Moving on to the next distributed setting...")
				break
	print(runtimes)
	plot(runtimes)
main()