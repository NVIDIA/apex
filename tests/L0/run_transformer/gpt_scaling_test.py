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


def set_seed():
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0)
	torch.cuda.manual_seed_all(0)


def run_gpt(n):
	cmd = 'python3 native.py ' + str(n)
	args = list(cmd.split(' '))
	try:
		p = subprocess.Popen(args)
		outs, errs = p.communicate()
	except:
		traceback.print_exc(file=sys.stdout)
		print(args)
		print("Inference failed")
	filename = 'native' + str(n) + '.txt'
	fileread = str(open(filename,'r').read())
	runtime = float(fileread)
	os.remove(filename)
	return runtime


def plot(runtimes):
	import matplotlib.pyplot as plt
	plt.scatter(runtimes.keys(), runtimes.values())
	plt.xlabel('Parameters (Billions)')
	plt.ylabel('Training Iteration time (s)')
	plt.title(str("Offload Foward Pass Scaling"))
	plt.savefig('offload_scaling.png')
	plt.close()


def main():
	runtimes = {}
	try:
		for n in range(500,1000000,500):
			bill_params = round((1024.0 * 1025.0 / (10.0**9)) * n,3)
			print("Testing", n,"1024x1024 layers ->", bill_params, 'billion parameters:', end=' ')
			runtime = run_offload(n)
			runtimes[bill_params] = runtime
			print(runtime, 'ms')
	except Exception as e:
		print(e)
		print(runtimes)
		plot(runtimes)
