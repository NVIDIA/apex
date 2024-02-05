import os
import timeit
import torch
import apex.contrib.gpu_direct_storage as gds

def run_benchmark(func):
    sizes = [2 ** i for i in range(16, 28)]
    for size in sizes:
        torch.cuda.empty_cache()
        s = torch.cuda.Stream()
        x = torch.linspace(0, 1, size, device = "cuda")

        # warmup
        torch.cuda.synchronize()
        for _ in range(10):
            func(x, f"{size}.data")
            os.remove(f"{size}.data")

        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        for _ in range(10):
            func(x, f"{size}.data")
            os.remove(f"{size}.data")
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        print(f"{func.__name__}: size = {size}, {end_time - start_time}")

if __name__ == '__main__':
    torch.cuda.set_device(0)
    run_benchmark(torch.save)
    run_benchmark(gds.save_data_no_gds)
    run_benchmark(gds.save_data)
