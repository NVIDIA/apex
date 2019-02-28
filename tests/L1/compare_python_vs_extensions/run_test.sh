#!/bin/bash

DATADIR="/home/mcarilli/Desktop/pt18data/apex/examples/imagenet/bare_metal_train_val/"
BASE_CMD="python main_amp.py -a resnet50 --b 128 --workers 4 --deterministic --prints-to-process 5"

print_banner() {
  printf "\n\n\n\e[30m\e[42m$1\e[0m\n\n\n\n"
}

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

pushd ../../..
python setup.py install --cuda_ext --cpp_ext
popd

for opt_level in "${opt_levels[@]}"
do
  for loss_scale in "${loss_scales[@]}"
  do
    for keep_batchnorm in "${keep_batchnorms[@]}"
    do
      print_banner "$BASE_CMD --opt-level $opt_level ${loss_scale} ${keep_batchnorm} --has-ext $DATADIR"
      set -x
      $BASE_CMD --opt-level $opt_level ${loss_scale} ${keep_batchnorm} --has-ext $DATADIR
      set +x
    done
  done
done

pushd ../../..
python setup.py install
popd

for opt_level in "${opt_levels[@]}"
do
  for loss_scale in "${loss_scales[@]}"
  do
    for keep_batchnorm in "${keep_batchnorms[@]}"
    do
      print_banner "$BASE_CMD --opt-level $opt_level ${loss_scale} ${keep_batchnorm} $DATADIR"
      set -x
      $BASE_CMD --opt-level $opt_level ${loss_scale} ${keep_batchnorm} $DATADIR
      set +x
    done
  done
done

for opt_level in "${opt_levels[@]}"
do
  for loss_scale in "${loss_scales[@]}"
  do
    for keep_batchnorm in "${keep_batchnorms[@]}"
    do
      set -x
      python compare.py --opt-level $opt_level ${loss_scale} ${keep_batchnorm}
      set +x
    done
  done
done

pushd ../../..
python setup.py install --cuda_ext --cpp_ext
popd
