import torch
import sys
import subprocess

def docstring_hack():
    """
    Multiproc file which will launch a set of processes locally for multi-gpu
    usage: python -m apex.parallel.multiproc main.py ...
    """
    pass

argslist = list(sys.argv)[1:]
world_size = torch.cuda.device_count()

if '--world-size' in argslist:
    world_size = int(argslist[argslist.index('--world-size')+1])
else:
    argslist.append('--world-size')
    argslist.append(str(world_size))

workers = []

for i in range(world_size):
    if '--rank' in argslist:
        argslist[argslist.index('--rank')+1] = str(i)
    else:
        argslist.append('--rank')
        argslist.append(str(i))
    stdout = None if i == 0 else open("GPU_"+str(i)+".log", "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)]+argslist, stdout=stdout)
    workers.append(p)

for p in workers:
    p.wait()
