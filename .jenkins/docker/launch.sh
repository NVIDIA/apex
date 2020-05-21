sudo docker run -it -v $HOME:/data --rm --privileged --device=/dev/dri --device=/dev/kfd --network host --group-add video apex
