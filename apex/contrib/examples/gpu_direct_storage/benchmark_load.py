import os
import timeit
import torch
import apex.contrib.gpu_direct_storage as gds

def run_benchmark_torch_load():
    sizes = [2 ** i for i in range(16, 28)]
    for size in sizes:
        torch.cuda.empty_cache()
        s = torch.cuda.Stream()
        x = torch.empty(size, device = "cuda")
        y = torch.linspace(0, 1, size, device = "cuda")
        torch.save(y, f"{size}.data")

        # warmup
        torch.cuda.synchronize()
        for _ in range(10):
            x = torch.load(f"{size}.data")

        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        for _ in range(10):
            x = torch.load(f"{size}.data")
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        print(f"torch.load: size = {size}, {end_time - start_time}")
        assert(torch.allclose(x, y))

def run_benchmark(func):
    sizes = [2 ** i for i in range(16, 28)]
    for size in sizes:
        torch.cuda.empty_cache()
        s = torch.cuda.Stream()
        x = torch.empty(size, device = "cuda")
        y = torch.linspace(0, 1, size, device = "cuda")

        with gds.GDSFile(f"{size}.data", "w") as f:
            f.save_data(y)

        # warmup
        torch.cuda.synchronize()
        for _ in range(10):
            func(x, f"{size}.data")

        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        for _ in range(10):
            func(x, f"{size}.data")
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        print(f"{func.__name__}: size = {size}, {end_time - start_time}")
        assert(torch.allclose(x, y))

def load_data_yes_gds(tensor, filename):
    with gds.GDSFile(filename, "r") as f:
        f.load_data(tensor)

def load_data_no_gds(tensor, filename):
    with gds.GDSFile(filename, "rn") as f:
        f.load_data_no_gds(tensor)

if __name__ == '__main__':
    run_benchmark_torch_load()
    run_benchmark(load_data_yes_gds)
    run_benchmark(load_data_no_gds)
