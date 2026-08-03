import apex.contrib.gpu_direct_storage as gds
import torch

for size in [128, 1024, 8192]:
    x = torch.linspace(0, 1, size, device = "cuda")
    with gds.GDSFile(f"{size}.data", "w") as f:
        f.save_data(x)
