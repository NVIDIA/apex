import subprocess
import os
from apex.transformer.testing.commons import TEST_SUCCESS_MESSAGE

def run_gpt(cmd):
	args = list(cmd.split(' '))
	p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	outs, errs = p.communicate()
	outs = list(str((outs).decode('utf-8')).splitlines())
	success = False
	runtime = 0
	num_params = 0
	for out in outs:
		out=str(out)
		if "Average Iteration Time:" in str(out):
			slicey = out[out.find(':')+2:]
			try:
				runtime = float(slicey)
			except:
				print(slicey)
				quit()
		if "Number of Parameters:" in str(out):
			slicey = out[out.find(':')+2:]
			try:
				num_params = int(slicey)
			except:
				print(slicey)
				quit()
		if str(out) == str(TEST_SUCCESS_MESSAGE):
			success=True
	return runtime, round(float(int(num_params))/10.0**9,3), success, errs


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
	if not os.path.exists('/my_workspace/'):
		os.system('mkdir /my_workspace/')
	os.system('cp *.png /my_workspace/')


def main():
	runtimes = {}
	nlist = list(range(2000,10000,2000)) + list(range(10000,50000,5000)) + list(range(50000,100000,10000))
	print("N-List:", nlist)
	for data_parr, tens_parr, pipe_parr in [(8,1,1), (4,2,1), (2,1,4), (1,2,4)]:
		for offload in [True, False]:
			dist_setting = 'ddp=' + str(data_parr) + ', tensor_parr=' + str(tens_parr) + ', pipe_parr=' + str(pipe_parr) + ', offload=' + str(offload)
			runtimes[dist_setting] = {} 
			print("Beginning Testing for", dist_setting)
			for n in nlist:
				cmd = "python3 -m torch.distributed.launch --nproc_per_node=8 run_gpt_minimal_test.py"
				cmd += " --micro-batch-size 1 --num-layers " + str(n) + " --hidden-size 128 --num-attention-heads 16"
				cmd += ' --max-position-embeddings 128 --seq-length 128 --tensor-model-parallel-size ' + str(tens_parr)
				cmd += " --pipeline-model-parallel-size " + str(pipe_parr) + (' --cpu-offload' if offload else '')
				print(cmd)
				runtime, bill_params, success, errs = run_gpt(cmd)
				if success:
					runtimes[dist_setting][bill_params] = runtime
					print(str(runtime) + 's per training iter for', str(bill_params) + 'B parameter GPT-2')
					if n >= 10000:
						plot(runtimes)
				else:
					print("GPT-2 w/", n, "layers failed using", dist_setting)
					print("Moving on to the next distributed setting...")
					print("#"*(25))
					print()
					plot(runtimes)
					break
	print(runtimes)
	plot(runtimes)
if __name__ == "__main__":
    main()