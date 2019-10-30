## Option 1:  Create a new container with Apex

**Dockerfile** installs the latest Apex on top of an existing image.  Run
```
docker build -t new_image_with_apex .
```
By default, **Dockerfile** uses NVIDIA's Pytorch container as the base image,
which requires an NVIDIA GPU Cloud (NGC) account.  If you don't have an NGC account, you can sign up for free by following the instructions [here](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html#generating-api-key).

Alternatively, you can supply your own base image via the `BASE_IMAGE` build-arg.
`BASE_IMAGE` must have Pytorch and Cuda installed.  For example, any
`-devel` image for Pytorch 1.0 and later from the
[official Pytorch Dockerhub](https://hub.docker.com/r/pytorch/pytorch) may be used:
```
docker build --build-arg BASE_IMAGE=1.3-cuda10.1-cudnn7-devel -t new_image_with_apex .
```

If you want to rebuild your image, and force the latest Apex to be cloned and installed, make any small change to the `SHA` variable in **Dockerfile**.

**Warning:**
Currently, the non-`-devel` images on Pytorch Dockerhub do not contain the Cuda compiler `nvcc`.  Therefore,
images whose name does not contain `-devel` are not eligible candidates for `BASE_IMAGE`.

### Running your Apex container

Like any Cuda-enabled Pytorch container, a container with Apex should be run via [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), for example:
```
docker run --runtime=nvidia -it --rm --ipc=host new_image_with_apex
```

## Option 2:  Install Apex in a running container

Instead of building a new container, it is also a viable option to `git clone https://github.com/NVIDIA/apex.git` on bare metal, mount the Apex repo into your container at launch by running, for example,
```
docker run --runtime=nvidia -it --rm --ipc=host -v /bare/metal/apex:/apex/in/container <base image>
```
then go to /apex/in/container within the running container and
```
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```
