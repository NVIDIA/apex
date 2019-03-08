## Option 1:  Create a new container with Apex

**Dockerfile** installs the latest Apex on top of an existing image.  Run
```
docker build -t image_with_apex .
```
By default, **Dockerfile** uses NVIDIA's Pytorch container as the base image,
which requires an NVIDIA GPU Cloud (NGC) account.  If you don't have an NGC account, you can sign up for free by following the instructions [here](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html#generating-api-key).

Alternatively, you can supply your own base image via the `BASE_IMAGE` build-arg.
Any `BASE_IMAGE` you supply must have Pytorch and Cuda installed, for example:
```
docker build --build-arg BASE_IMAGE=pytorch/pytorch:0.4-cuda9-cudnn7-devel -t image_with_apex .
```

If you want to rebuild your image, and force the latest Apex to be cloned and installed, make any small change to the `SHA` variable in **Dockerfile**.

**Warning:**
Currently, Pytorch's default non-devel image on Dockerhub
[pytorch/pytorch:0.4_cuda9_cudnn7](https://hub.docker.com/r/pytorch/pytorch/tags/) contains Pytorch installed with prebuilt binaries.  It does not contain NVCC, which means it is not an eligible candidate for `<base image>`.

## Option 2:  Install Apex in a running container

Instead of building a new container, it is also a viable option to `git clone https://github.com/NVIDIA/apex.git` on bare metal, mount the Apex repo into your container at launch by running, for example,
```
docker run --runtime=nvidia -it --rm --ipc=host -v /bare/metal/apex:/apex/in/container <base image>
```
then go to /apex/in/container within the running container and `python setup.py install [--cuda_ext] [--cpp_ext]`.
