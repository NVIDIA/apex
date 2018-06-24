# Base image must at least have nvcc and CUDA installed.
# FROM gitlab-dl.nvidia.com:5005/dgx/pytorch:18.04-py3-devel
# FROM gitlab-dl.nvidia.com:5005/dgx/pytorch:18.04-py3-devel
FROM pytorch/pytorch:0.4_cuda9_cudnn7
WORKDIR /workspace
# uninstall Apex if present
RUN pip uninstall -y apex || :
# We'd like Docker to clone a fresh version of Apex each time.
# However, if the RUN line below does not change from the last
# time you built this image, Docker won't rebuild the layer.
# SHA is a manual hack, something the user can change to touch
# the line and force Docker to recreate the layer (which ensures
# that Docker actually runs the "git clone").
RUN SHA=tOuChMe git clone file:///home/mcarilli/Desktop/pt18data/apex
WORKDIR /workspace/apex
RUN python setup.py install
WORKDIR /workspace
