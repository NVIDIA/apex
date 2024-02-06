import torch
import apex.contrib.gpu_direct_storage as gds

torch.cuda.set_device(0)

for size in [128, 1024, 8192]:
    x = torch.empty(size, device = "cuda")
    gds.load_data(x, f"{size}.data")
    xx = torch.linspace(0, 1, size, device = "cuda")
    assert(torch.allclose(x, xx))
