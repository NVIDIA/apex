#!/bin/bash

print_banner() {
  printf "\n\n\n\e[30m\e[42m$1\e[0m\n\n\n\n"
}

print_banner "Distributed status:  $1"

echo $2
DATADIR=$2

if [ -n "$3" ]
then
  USE_BASELINE=""
else
  USE_BASELINE="--use_baseline"
fi

if [ "$1" == "single_gpu" ]
then
  BASE_CMD="python main_amp.py -a resnet50 --b 128 --workers 4 --deterministic --prints-to-process 5"
fi

if [ "$1" == "distributed" ]
then
  BASE_CMD="python -m torch.distributed.launch --nproc_per_node=2 main_amp.py -a resnet50 --b 128 --workers 4 --deterministic --prints-to-process 5"
fi

ADAM_ARGS="--opt-level O2 --keep-batchnorm-fp32 False --fused-adam"

keep_batchnorms=(
""
"--keep-batchnorm-fp32 True"
"--keep-batchnorm-fp32 False"
)

loss_scales=(
""
"--loss-scale 1.0"
"--loss-scale 128.0"
"--loss-scale dynamic"
)

opt_levels=(
"O0"
"O1"
"O2"
"O3"
)

rm True*
rm False*

set -e

print_banner "Installing Apex with --cuda_ext and --cpp_ext"

pushd ../../..
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
popd

for opt_level in "${opt_levels[@]}"
do
  for loss_scale in "${loss_scales[@]}"
  do
    for keep_batchnorm in "${keep_batchnorms[@]}"
    do
      if [ "$opt_level" == "O1" ] && [ -n "${keep_batchnorm}" ]
      then
        print_banner "Skipping ${opt_level} ${loss_scale} ${keep_batchnorm}"
        continue
      fi
      print_banner "${BASE_CMD} --opt-level ${opt_level} ${loss_scale} ${keep_batchnorm} --has-ext $DATADIR"
      set -x
      ${BASE_CMD} --opt-level ${opt_level} ${loss_scale} ${keep_batchnorm} --has-ext $DATADIR
      set +x
    done
  done
done

# Handle FusedAdam separately due to limited support.
# FusedAdam will not be tested for bitwise accuracy against the Python implementation.
# The L0 tests already do so.  These tests are here to ensure that it actually runs,
# and get an idea of performance.
for loss_scale in "${loss_scales[@]}"
do
  print_banner "${BASE_CMD} ${ADAM_ARGS} ${loss_scale} --has-ext $DATADIR"
  set -x
  ${BASE_CMD} ${ADAM_ARGS} ${loss_scale} --has-ext $DATADIR
  set +x
done

print_banner "Reinstalling apex without extensions"

pushd ../../..
pip install -v --no-cache-dir .
popd

for opt_level in "${opt_levels[@]}"
do
  for loss_scale in "${loss_scales[@]}"
  do
    for keep_batchnorm in "${keep_batchnorms[@]}"
    do
      if [ "$opt_level" == "O1" ] && [ -n "${keep_batchnorm}" ]
      then
        print_banner "Skipping ${opt_level} ${loss_scale} ${keep_batchnorm}"
        continue
      fi
      print_banner "${BASE_CMD} --opt-level ${opt_level} ${loss_scale} ${keep_batchnorm} $DATADIR"
      set -x
      ${BASE_CMD} --opt-level ${opt_level} ${loss_scale} ${keep_batchnorm} $DATADIR
      set +x
    done
  done
done

print_banner "Checking for bitwise accuracy between Python-only and cpp/cuda extension installs"

for opt_level in "${opt_levels[@]}"
do
  for loss_scale in "${loss_scales[@]}"
  do
    for keep_batchnorm in "${keep_batchnorms[@]}"
    do
      echo ""
      if [ "$opt_level" == "O1" ] && [ -n "${keep_batchnorm}" ]
      then
        echo "Skipping ${opt_level} ${loss_scale} ${keep_batchnorm}"
        continue
      fi
      echo "${BASE_CMD} --opt-level ${opt_level} ${loss_scale} ${keep_batchnorm} [--has-ext] $DATADIR"
      set -x
      python compare.py --opt-level ${opt_level} ${loss_scale} ${keep_batchnorm} --use_baseline
      set +x
    done
  done
done

print_banner "Reinstalling Apex with --cuda_ext and --cpp_ext"

pushd ../../..
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
popd
