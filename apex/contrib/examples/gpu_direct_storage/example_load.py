import torch
import apex.contrib.gpu_direct_storage as gds

for size in [128, 1024, 8192]:
    x = torch.empty(size, device = "cuda")
    with gds.GDSFile(f"{size}.data", "r") as f:
        f.load_data(x)
    xx = torch.linspace(0, 1, size, device = "cuda")
    assert(torch.allclose(x, xx))
