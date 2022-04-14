set -ex
ROOT_DIR=`dirname $0`

nvcc ${ROOT_DIR}/test_fmha_so.cc -o ${ROOT_DIR}/test_fmha_so -lfmha -ldl
${ROOT_DIR}/test_fmha_so
