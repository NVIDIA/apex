import sys

import torch
from torch import distributed as dist
from torch.testing._internal import common_utils
from torch.testing._internal import common_distributed


class DistributedTestBase(common_distributed.MultiProcessTestCase):

    BACKEND_NCCL = "nccl"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        super().tearDown()

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 4)

    @property
    def init_method(self):
        return f"{common_utils.FILE_SCHEMA}{self.file_name}"

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.assertTrue(torch.cuda.is_available())
        self.rank = rank
        self.file_name = file_name

        print(f"[dist init] rank = {self.rank}, world_size = {self.world_size}")

        try:
            dist.init_process_group(
                init_method=self.init_method,
                backend=DistributedTestBase.BACKEND_NCCL,
                world_size=int(self.world_size),
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                print(f"Backend of {DistributedTestBase.BACKEND_NCCL} not available")
                sys.exit(0)
            raise

        torch.cuda.set_device(self.rank % torch.cuda.device_count())

        dist.barrier()
        self.run_test(test_name, pipe)
        dist.barrier()

        dist.destroy_process_group()
        sys.exit(0)
