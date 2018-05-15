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

def find(path, regex_func, collect=False):
    """
    Recursively searches through a directory with regex_func and
    either collects all instances or returns the first instance.
    
    Args:
        path: Directory to search through
        regex_function: A function to run on each file to decide if it should be returned/collected
        collect (False) : If True will collect all instances of matching, else will return first instance only 
    """
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
    """
    Based on PyTorch build process. Will look for nvcc for compilation.
    Either will set cuda home by enviornment variable CUDA_HOME or will search
    for nvcc. Returns NVCC executable, cuda major version and cuda home directory.
    """
    cuda_path = None
    CUDA_HOME = None

    CUDA_HOME = os.getenv('CUDA_HOME', '/usr/local/cuda')
    if not os.path.exists(CUDA_HOME):
        # We use nvcc path on Linux and cudart path on macOS
        cudart_path = ctypes.util.find_library('cudart')
        if cudart_path is not None:
            cuda_path = os.path.dirname(cudart_path)
        if cuda_path is not None:
            CUDA_HOME = os.path.dirname(cuda_path)
            
    if not cuda_path and not CUDA_HOME:
        nvcc_path = find('/usr/local/', re.compile("nvcc").search, False)
        if nvcc_path:
            CUDA_HOME = os.path.dirname(nvcc_path)
            if CUDA_HOME:
                os.path.dirname(CUDA_HOME)

        if (not os.path.exists(CUDA_HOME+os.sep+"lib64")
            or not os.path.exists(CUDA_HOME+os.sep+"include") ):
            raise RuntimeError("Error: found NVCC at ", nvcc_path ," but could not locate CUDA libraries"+
                               " or include directories.")
        
        raise RuntimeError("Error: Could not find cuda on this system."+
                           " Please set your CUDA_HOME enviornment variable to the CUDA base directory.")
    
    NVCC = find(CUDA_HOME+os.sep+"bin", 
                re.compile('nvcc$').search)
    print("Found NVCC = ", NVCC)

    # Parse output of nvcc to get cuda major version
    nvcc_output = subprocess.check_output([NVCC, '--version']).decode("utf-8")
    CUDA_LIB = re.compile(', V[0-9]+\.[0-9]+\.[0-9]+').search(nvcc_output).group(0).split('V')[1]
    print("Found CUDA_LIB = ", CUDA_LIB)

    if CUDA_LIB:
        try:
            CUDA_VERSION = int(CUDA_LIB.split('.')[0])
        except (ValueError, TypeError):
            CUDA_VERSION = 9
    else:
         CUDA_VERSION = 9   

    if CUDA_VERSION < 8:
        raise RuntimeError("Error: APEx requires CUDA 8 or newer")

    
    return NVCC, CUDA_VERSION, CUDA_HOME

#Get some important paths
curdir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
buildir = curdir+os.sep+"build"
if not os.path.exists(buildir):
    os.makedirs(buildir)

torch_dir = os.path.split(torch.__file__)[0] + os.sep + "lib"

cuda_files = find(curdir, lambda file: file.endswith(".cu"), True)
cuda_headers = find(curdir, lambda file: file.endswith(".cuh"), True)
headers = find(curdir, lambda file: file.endswith(".h"), True)

libaten = list(set(find(torch_dir, re.compile("libaten", re.IGNORECASE).search, True)))
libaten_names = [os.path.splitext(os.path.basename(entry))[0] for entry in libaten]
for i, entry in enumerate(libaten_names):
    if entry[:3]=='lib':
        libaten_names[i] = entry[3:]

aten_h = find(torch_dir, re.compile("aten.h", re.IGNORECASE).search, False)

include_dirs = [os.path.dirname(os.path.dirname(aten_h))]
library_dirs = []
for file in cuda_headers+headers:
    dir = os.path.dirname(file)
    if dir not in include_dirs:
        include_dirs.append(dir)

assert libaten, "Could not find PyTorch's libATen."
assert aten_h, "Could not find PyTorch's ATen header."

library_dirs.append(os.path.dirname(libaten[0]))

#create some places to collect important things
object_files = []
extra_link_args=[]
main_libraries = []
main_libraries += ['cudart',]+libaten_names
extra_compile_args = ["--std=c++11",]

#findcuda returns root dir of CUDA
#include cuda/include and cuda/lib64 for python module build.
NVCC, CUDA_VERSION, CUDA_HOME=findcuda()
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

def CompileCudaFiles(NVCC, CUDA_VERSION):

        print()
        print("Compiling cuda modules with nvcc:")
        gencodes =  ['-gencode', 'arch=compute_52,code=sm_52',
                    '-gencode', 'arch=compute_60,code=sm_60',
                    '-gencode', 'arch=compute_61,code=sm_61',]
        
        if CUDA_VERSION > 8:
            gencodes += ['-gencode', 'arch=compute_70,code=sm_70',
                         '-gencode', 'arch=compute_70,code=compute_70',]
        #Need arches to compile for. Compiles for 70 which requires CUDA9
        nvcc_cmd = [NVCC, 
                       '-Xcompiler', 
                       '-fPIC'
                   ] + gencodes + [
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
    CompileCudaFiles(NVCC, CUDA_VERSION)

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
