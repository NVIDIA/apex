1. How do I intercept the Adam optimizer in APEX ?

	```python
	from apex import pyprof
	import fused_adam_cuda
	pyprof.nvtx.wrap(fused_adam_cuda, 'adam')
	```

2. If you are using JIT and/or AMP, the correct initialization sequence is
	1. Let any JIT to finish.
	2. Initlialize pyprof `pyprof.nvtx.init()`.
	3. Initialize AMP.

3. How do I profile with `torch.distributed.launch` ?

	```python
	nvprof -f -o net%p.sql \
		--profile-from-start off \
		--profile-child-processes \
		python -m torch.distributed.launch net.py
	```
