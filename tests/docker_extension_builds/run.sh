#!/bin/bash

print_banner() {
  printf "\n\n\n\e[30m\e[42m$1\e[0m\n\n\n\n"
}

print_green() {
  printf "\e[30m\e[42m$1\e[0m\n"
}

print_red() {
  printf "\e[30m\e[41m$1\e[0m\n"
}

images=(
"gitlab-master.nvidia.com:5005/dl/dgx/pytorch:19.08-py3-devel"
"gitlab-master.nvidia.com:5005/dl/dgx/pytorch:master-py3-devel"
"pytorch/pytorch:nightly-devel-cuda10.0-cudnn7"
"pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel"
"pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel"
"pytorch/pytorch:1.0-cuda10.0-cudnn7-devel"
"pytorch/pytorch:nightly-devel-cuda9.2-cudnn7"
)

branch="master"

# Associative array for exit codes
declare -A exit_codes
for image in images
do
  exit_codes[$image]="None"
done

for image in "${images[@]}"
do
  print_banner "$image"
  set -x
  docker pull $image
  # Trying python setup.py install instead of pip install to ensure direct access to error codes.
  # Maybe pip install would be ok too but this works.
  docker run --runtime=nvidia --rm $image /bin/bash -c "yes | pip uninstall apex; yes | pip uninstall apex; git clone https://github.com/NVIDIA/apex.git; cd apex; git checkout $branch; set -e;  python setup.py install --cuda_ext --cpp_ext"
  exit_code=$?
  set +x
  if [ $exit_code != 0 ]
  then
    print_red "Exit code: $exit_code"
  else
    print_green "Exit code: $exit_code"
  fi
  exit_codes[$image]=$exit_code
done

success=0
for image in "${images[@]}"
do
  exit_code=${exit_codes[$image]}
  if [ $exit_code != 0 ]
  then
    print_red "$image : $exit_code"
    success=1
  else
    print_green "$image : $exit_code"
  fi
done

if [ $success != 0 ]
then
  print_red "Overall status:  failure"
else
  print_green "Overall status:  success"
fi

exit $success
