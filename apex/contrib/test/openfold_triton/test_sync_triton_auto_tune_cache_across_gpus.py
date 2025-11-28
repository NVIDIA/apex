import os

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)

from apex.contrib.openfold_triton import (
    LayerNormSmallShapeOptImpl,
    sync_triton_auto_tune_cache_across_gpus,
    _tuneable_triton_kernels,
)


class SyncTritonAutoTuneCacheTest(MultiProcessTestCase):
    device_type = "cuda"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        super().tearDown()

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    @property
    def init_method(self):
        return f"{common_utils.FILE_SCHEMA}{self.file_name}"

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return True

    def _create_process_group_nccl(self):
        def maybe_export(env, val):
            if not type(env) == str:
                raise ValueError(f"Type of type of env is expected to be str, but got {type(env)}")
            if not type(val) == str:
                raise ValueError(f"Type of type of val is expected to be str, but got {type(val)}")
            if os.getenv(env) is None:
                os.environ[env] = val

        maybe_export("MASTER_PORT", "29500")
        maybe_export("MASTER_ADDR", "localhost")

        # create nccl processgroup for two ranks
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
        )
        pg = dist.distributed_c10d._get_default_group()
        return pg

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_sync_triton_auto_tune_cache_across_gpus(self):
        pg = self._create_process_group_nccl()
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)

        if self.rank == 0:
            eps = 1e-5
            normalized_shape = (
                128,
                64,
            )

            weight = torch.ones(normalized_shape, device=device, requires_grad=True)
            bias = torch.zeros(normalized_shape, device=device, requires_grad=True)

            x = torch.randn(
                (
                    2,
                    2,
                )
                + normalized_shape,
                device=device,
            )
            y = LayerNormSmallShapeOptImpl.apply(x, normalized_shape, weight, bias, eps)
            l = torch.sum(y)
            l.backward()

        sync_triton_auto_tune_cache_across_gpus(strict=False, verbose=True)

        caches_synced = 0
        for func_name, func in _tuneable_triton_kernels.items():
            if len(func.cache) > 0:
                caches_synced = caches_synced + 1
                print(
                    f"caches were synchronized for {func_name} at rank = {self.rank}:",
                    func.cache,
                )

        self.assertTrue(caches_synced > 0)


if __name__ == "__main__":
    run_tests()
