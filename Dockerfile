ARG FROM_IMAGE=lcskrishna/rocm-pytorch:rocm3.3_ubuntu16.04_py3.6_pytorch_bfloat16_mgpu

FROM ${FROM_IMAGE}
RUN \
    git clone --recursive https://github.com/ROCmSoftwarePlatform/apex.git && \
    cd apex && \
    python3.6 setup.py install --cpp_ext --cuda_ext
