**Dockerfile** is a simple template that shows how to install the latest Apex on top of an existing image.  Edit **Dockerfile** to choose a base image, then run 
```
docker build -t image_with_apex ."
```
.  If you want to rebuild your image, and force the latest Apex to be cloned and installed, make any small change to the `SHA` variable on line 8.

**base_images.md** provides guidance on base images to use in the `FROM <base image>` line of **Dockerfile**.

Instead of building a new container, it is also a viable option to clone Apex on bare metal, mount the Apex repo into your container at launch by running, for example,
```
docker run --runtime=nvidia -it --rm --ipc=host -v /bare/metal/apex:/apex/in/container <base image>
```
, then go to /apex/in/container within the running container and `python setup.py install`.
