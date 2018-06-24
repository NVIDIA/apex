When specifying 
```
FROM <base image>
```
in *Dockerfile*, `<base image>` must have Pytorch and CUDA installed.

If you have an NGC account, you can use Nvidia's official Pytorch container
```
nvcr.io/nvidia/pytorch:18.04-py3
```
as `<base image>`.
If you don't have an NGC account, you can sign up for one for free by following the instructions [here](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html#generating-api-key).

An alternative is to first 
[build a local Pytorch image](https://github.com/pytorch/pytorch#docker-image) using Pytorch's Dockerfile on Github. From the root of your cloned Pytorch repo,
run
```
docker build -t my_pytorch_image -f docker/pytorch/Dockerfile .
```
`my_pytorch_image` will contain CUDA, and can be used as `<base image>`.

Warning:
Currently, Pytorch's latest stable image on Dockerhub
[pytorch/pytorch:0.4_cuda9_cudnn7](https://hub.docker.com/r/pytorch/pytorch/tags/) contains Pytorch installed with prebuilt binaries.  It does not contain NVCC, which means it is not an eligible candidate for `<base image>`.
