import re

import sys
import os
import shutil
import inspect

import distutils
import distutils.spawn
from distutils.command.clean import clean

from setuptools import setup, Extension, find_packages
from setuptools.command.install import install

import subprocess
import ctypes.util

import torch

#Takes a path to walk
#A function to decide if to keep
#collection if we want a list of all occurances
def find(path, regex_func, collect=False):
    collection = [] if collect else None
    for root, dirs, files in os.walk(path):
        for file in files:
            if regex_func(file):
                if collect:
                    collection.append(os.path.join(root, file))
                else:
                    return os.path.join(root, file)
    return list(set(collection))

def findcuda():
    CUDA_HOME = os.getenv('CUDA_HOME', '/usr/local/cuda')
    if not os.path.exists(CUDA_HOME):
        # We use nvcc path on Linux and cudart path on macOS
        osname = platform.system()
        if osname == 'Linux':
            cuda_path = find_nvcc()
        else:
            cudart_path = ctypes.util.find_library('cudart')
            if cudart_path is not None:
                cuda_path = os.path.dirname(cudart_path)
            else:
                cuda_path = None
        if cuda_path is not None:
            CUDA_HOME = os.path.dirname(cuda_path)
        else:
            CUDA_HOME = None
    WITH_CUDA = CUDA_HOME is not None
    return CUDA_HOME

#Get some important paths
curdir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
buildir = curdir+os.sep+"build"
if not os.path.exists(buildir):
    os.makedirs(buildir)

torch_dir = os.path.split(torch.__file__)[0] + os.sep + "lib"

cuda_files = find(curdir, lambda file: file.endswith(".cu"), True)
cuda_headers = find(curdir, lambda file: file.endswith(".cuh"), True)
headers = find(curdir, lambda file: file.endswith(".h"), True)

libaten = find(torch_dir, re.compile("libaten", re.IGNORECASE).search, False)
aten_h = find(torch_dir, re.compile("aten.h", re.IGNORECASE).search, False)

include_dirs = [os.path.dirname(os.path.dirname(aten_h))]
library_dirs = []
for file in cuda_headers+headers:
    dir = os.path.dirname(file)
    if dir not in include_dirs:
        include_dirs.append(dir)

assert libaten, "Could not find PyTorch's libATen."
assert aten_h, "Could not find PyTorch's ATen header."

library_dirs.append(os.path.dirname(libaten))

#create some places to collect important things
object_files = []
extra_link_args=[]
main_libraries = []
main_libraries += ['cudart', 'cuda', 'ATen']
extra_compile_args = ["--std=c++11",]

#findcuda returns root dir of CUDA
#include cuda/include and cuda/lib64 for python module build.
CUDA_HOME=findcuda()
library_dirs.append(os.path.join(CUDA_HOME, "lib64"))
include_dirs.append(os.path.join(CUDA_HOME, 'include'))

class RMBuild(clean):
    def run(self):
        #BE VERY CAUTIOUS WHEN USING RMTREE!!!
        #These are some carefully written/crafted directories
        if os.path.exists(buildir):
            shutil.rmtree(buildir)
            
        distdir = curdir+os.sep+"dist"
        if os.path.exists(distdir):
            shutil.rmtree(distdir)

        eggdir = curdir+os.sep+"apex.egg-info"
        if os.path.exists(eggdir):
            shutil.rmtree(eggdir)
        clean.run(self)

def CompileCudaFiles():

        print()
        print("Compiling cuda modules with nvcc:")
        #Need arches to compile for. Compiles for 70 which requires CUDA9
        nvcc_cmd = ['nvcc', 
                    '-Xcompiler', 
                    '-fPIC',
                    '-gencode', 'arch=compute_52,code=sm_52',
                    '-gencode', 'arch=compute_60,code=sm_60',
                    '-gencode', 'arch=compute_61,code=sm_61',
                    '-gencode', 'arch=compute_70,code=sm_70',
                    '-gencode', 'arch=compute_70,code=compute_70',
                    '--std=c++11',
                    '-O3',
        ]
        
        for dir in include_dirs:
            nvcc_cmd.append("-I"+dir)

        for file in cuda_files:
            object_name = os.path.basename(
                os.path.splitext(file)[0]+".o"
            )
    
            object_file = os.path.join(buildir, object_name)
            object_files.append(object_file)
    
            file_opts = ['-c', file, '-o', object_file]
            
            print(' '.join(nvcc_cmd+file_opts))
            subprocess.check_call(nvcc_cmd+file_opts)
            
        for object_file in object_files:
            extra_link_args.append(object_file)
if 'clean' not in sys.argv:

    print()
    print("Arguments used to build CUDA extension:")
    print("extra_compile_args :", extra_compile_args)
    print("include_dirs: ", include_dirs)
    print("extra_link_args: ", extra_link_args)
    print("library_dirs: ", library_dirs)
    print("libraries: ", main_libraries)
    print()
    CompileCudaFiles()

    print("Building CUDA extension.")
    
cuda_ext = Extension('apex._C',
                 [os.path.join('csrc', 'Module.cpp')],
                 extra_compile_args = extra_compile_args,
                 include_dirs=include_dirs,
                 extra_link_args=extra_link_args,
                 library_dirs=library_dirs,
                 runtime_library_dirs = library_dirs,
                 libraries=main_libraries
    )

if 'clean' not in sys.argv:
    print("Building module.")
    
setup(
    name='apex', version='0.1',
    cmdclass={
        'clean' : RMBuild,
    },  
    ext_modules=[cuda_ext,],
    description='PyTorch Extensions written by NVIDIA',
    packages=find_packages(exclude=("build", "csrc", "include", "tests")),
)
