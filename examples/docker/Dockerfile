# Base image must at least have nvcc and CUDA installed.
FROM nvcr.io/nvidia/pytorch:18.04-py3
WORKDIR /workspace
# uninstall Apex if present
RUN pip uninstall -y apex || :
# SHA is something the user can alter to force recreation of this Docker layer, 
# and therefore force cloning the latest version of Apex
RUN SHA=43f1ae08 git clone https://github.com/NVIDIA/apex.git
WORKDIR /workspace/apex
RUN python setup.py install
WORKDIR /workspace
