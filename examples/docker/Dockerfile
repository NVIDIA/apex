# Base image must at least have pytorch and CUDA installed.
FROM <base image>
WORKDIR /workspace
# uninstall Apex if present
RUN pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer, 
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /workspace/apex
RUN python setup.py install
WORKDIR /workspace
