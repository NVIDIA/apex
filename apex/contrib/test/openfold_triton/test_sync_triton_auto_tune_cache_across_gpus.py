import os
from apex.contrib.openfold_triton import LayerNormSmallShapeOptImpl, sync_triton_auto_tune_cache_across_gpus, _tuneable_triton_kernels


import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase, requires_nccl, skip_if_lt_x_gpu

class SyncTritonAutoTuneCacheTest(MultiProcessTestCase):
    device_type = "cuda"
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        super().setUp()
        self._setup_pre_spawn()
        self._spawn_processes()

    def tearDown(self) -> None:
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
        # Overriding base test class: do not auto destroy PG upon exit.
        return True

    def _setup_pre_spawn(self):
        pass

    def _create_process_group_nccl(self):
        def maybe_export(env, val):
            assert(type(env) == str)
            assert(type(val) == str)
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
            normalized_shape = (128, 64,)

            weight = torch.ones(normalized_shape, device=device)
            bias= torch.zeros(normalized_shape, device=device)

            x = torch.randn((2, 2,) + normalized_shape, device=device)
            y = LayerNormSmallShapeOptImpl.apply(
                x, normalized_shape, weight, bias, eps
            )

        sync_triton_auto_tune_cache_across_gpus(strict = False)

        caches_were_synced = False
        for func_name, func in _tuneable_triton_kernels.items():
            if len(func.cache) > 0:
                print(f"caches were synced for {func_name} at rank = {self.rank}:", func.cache)
                caches_were_synced = True

        self.assertTrue(caches_were_synced)

