# APEX GPUDirect Storage

This module aims to add a PyTorch extension for [GPUDirect Storage](https://developer.nvidia.com/blog/gpudirect-storage/) (GDS) support through utilizing the [cuFile](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) library.

# Build command
```
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--gpu_direct_storage" ./
```

Alternatively:
```
python setup.py install --gpu_direct_storage
```

Check installation:
```
python -c "import torch; import apex.contrib.gpu_direct_storage"
```
