Nvidia NVTX range markers (https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) 
are a useful tool to capture and observe events and code ranges etc. 
Using PyTorch APIs e.g, `torch.cuda.nvtx.range_push("xxx")` and `torch.cuda.nvtx.range_pop()` users can easily add their own NVTX range markers. These markers can then be observed in the Nvidia Visual Profiler (NVVP).

While inserting NVTX markers (strings), if the users follow a specific string pattern `"layer:your_string_here"` e.g. `"layer:conv1"` or `"layer:encoder_layer_3_self_attention`, then `pyprof` will display the strings `conv1` and `encoder_layer_3_self_attention` next to the associated kernels in the output of `prof.py` when used with the `-c layer` option.

NVTX range markers can be nested and if users follow the above string pattern, the output of `prof.py` will show all the markers associated with a kernel.

The file `resnet.py` (a simplified version of the torchvision model) shows an example of how users can add (nested) NVTX markers with information which can greatly aid in understanding and analysis of networks.

Note that the pattern `"layer:your_string_here"` was chosen to aid information extraction by `pyprof`. The tool will work seamlessly even if there are other markers or no markers at all.

### To run

```sh
nvprof -fo resnet.sql --profile-from-start off python resnet.py
parse.py resnet.sql > resnet.dict
prof.py --csv -c idx,layer,dir,mod,op,kernel,params,sil resnet.dict
```

The file `resnet.sql` can also be opened with NVVP as usual.
