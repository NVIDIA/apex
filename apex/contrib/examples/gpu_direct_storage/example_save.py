import torch
import apex.contrib.gpu_direct_storage as gds

torch.cuda.set_device(0)

for size in [128, 1024, 8192]:
  x = torch.linspace(0, 1, size, device = "cuda")
  gds.save_data(x, f"{size}.data")
