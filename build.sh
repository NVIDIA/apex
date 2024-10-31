#!/bin/bash -x

export PYTORCH_ROCM_ARCH=gfx942
# export TENSILE_DB=0x40
# export HIPBLASLT_LOG_MASK=0xff


python setup.py develop --cuda_ext --cpp_ext
cp build/lib.linux-x86_64-cpython-39/fused_dense_cuda.cpython-39-x86_64-linux-gnu.so /opt/conda/envs/py_3.9/lib/python3.9/site-packages/.

# export HIPBLASLT_LOG_FILE=hipblaslt_bgrad.log

# python apex/contrib/test/fused_dense/test_fused_dense_1.py

# python apex/contrib/test/fused_dense/test_half_T.py
# python apex/contrib/test/fused_dense/test_half_NT.py
