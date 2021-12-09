set -ex

ROOT_DIR=`dirname "$0"`/..

function build_nvcc_obj() {
    nvcc -c $1 \
        -O3 \
	-Xcompiler="-fPIC" \
	-gencode arch=compute_80,code=sm_80 \
	-U__CUDA_NO_HALF_OPERATORS__ \
	-U__CUDA_NO_HALF_CONVERSIONS__ \
	-I${ROOT_DIR}/apex/contrib/csrc/ \
	-I${ROOT_DIR}/apex/contrib/csrc/fmha/src \
	-I${ROOT_DIR}/build_scripts \
	--expt-relaxed-constexpr \
	--expt-extended-lambda \
	--use_fast_math \
	-DVERSION_GE_1_1 \
	-DVERSION_GE_1_3 \
	-DVERSION_GE_1_5
}

APEX_CU_SRC_DIR=${ROOT_DIR}/apex/contrib/csrc/fmha/src
build_nvcc_obj ${APEX_CU_SRC_DIR}/../fmha_api.cpp
build_nvcc_obj $APEX_CU_SRC_DIR/fmha_noloop_reduce.cu
build_nvcc_obj $APEX_CU_SRC_DIR/fmha_fprop_fp16_128_64_kernel.sm80.cu
build_nvcc_obj $APEX_CU_SRC_DIR/fmha_fprop_fp16_256_64_kernel.sm80.cu
build_nvcc_obj $APEX_CU_SRC_DIR/fmha_fprop_fp16_384_64_kernel.sm80.cu
build_nvcc_obj $APEX_CU_SRC_DIR/fmha_fprop_fp16_512_64_kernel.sm80.cu
build_nvcc_obj $APEX_CU_SRC_DIR/fmha_dgrad_fp16_128_64_kernel.sm80.cu
build_nvcc_obj $APEX_CU_SRC_DIR/fmha_dgrad_fp16_256_64_kernel.sm80.cu
build_nvcc_obj $APEX_CU_SRC_DIR/fmha_dgrad_fp16_384_64_kernel.sm80.cu
build_nvcc_obj $APEX_CU_SRC_DIR/fmha_dgrad_fp16_512_64_kernel.sm80.cu

rm -rf *.so
nvcc -shared -Xcompiler="-fPIC" -o libfmha.so *.o 
rm -rf *.o
