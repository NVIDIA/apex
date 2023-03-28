import os
import sys
import unittest
from packaging.version import Version, parse

import torch
from torch import distributed as dist
from torch.utils import collect_env
from torch.testing._internal import common_utils
from torch.testing._internal import common_distributed

from apex.transformer._ucc_util import HAS_UCC

# NOTE(mkozuki): Version guard for ucc. ref: https://github.com/openucx/ucc/issues/496
_TORCH_UCC_COMPAT_NVIDIA_DRIVER_VERSION = Version("470.42.01")
_driver_version = None
if torch.cuda.is_available():
    _driver_version = parse(collect_env.get_nvidia_driver_version(collect_env.run))
HAS_TORCH_UCC_COMPAT_NVIDIA_DRIVER = _driver_version is not None and _driver_version >= _TORCH_UCC_COMPAT_NVIDIA_DRIVER_VERSION


class DistributedTestBase(common_distributed.MultiProcessTestCase):

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
        return min(torch.cuda.device_count(), 4)

    @property
    def init_method(self):
        return f"{common_utils.FILE_SCHEMA}{self.file_name}"

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.assertTrue(torch.cuda.is_available())
        self.assertTrue(hasattr(self, "DISTRIBUTED_BACKEND"))
        self.rank = rank
        self.file_name = file_name

        print(f"[dist init] rank = {self.rank}, world_size = {self.world_size}")

        try:
            dist.init_process_group(
                init_method=self.init_method,
                backend=self.DISTRIBUTED_BACKEND,
                world_size=int(self.world_size),
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                print(f"Backend of {self.DISTRIBUTED_BACKEND} not available")
                sys.exit(0)
            raise

        torch.cuda.set_device(self.rank % torch.cuda.device_count())

        dist.barrier()
        self.run_test(test_name, pipe)
        dist.barrier()

        dist.destroy_process_group()
        sys.exit(0)

    def _setup_pre_spawn(self):
        pass


class NcclDistributedTestBase(DistributedTestBase):

    DISTRIBUTED_BACKEND = "nccl"

@unittest.skipUnless(
    HAS_UCC,
    "Requires either torch ucc or pytorch build from source with native ucc installed and enabled",
)
@unittest.skipUnless(
    HAS_TORCH_UCC_COMPAT_NVIDIA_DRIVER,
    f"`torch_ucc` requires NVIDIA driver >= {_TORCH_UCC_COMPAT_NVIDIA_DRIVER_VERSION} but {_driver_version} found. "
    "See https://github.com/openucx/ucc/issues/496",
)

class UccDistributedTestBase(DistributedTestBase):

    DISTRIBUTED_BACKEND = "ucc"

    def _setup_pre_spawn(self) -> None:
        self.master_addr = "localhost"
        os.environ["MASTER_ADDR"] = "localhost"
        self._has_master_port = "MASTER_PORT" in os.environ
        if self._has_master_port:
            self.master_port = os.environ["MASTER_PORT"]
        else:
            try:
                from caffe2.torch.fb.common.utils import get_free_port
                self.master_port = str(get_free_port())
            except ImportError:
                self.master_port = "12375"
            os.environ["MASTER_PORT"] = self.master_port

        self._has_ucx_tls = "UCX_TLS" in os.environ
        if not self._has_ucx_tls:
            os.environ["UCX_TLS"] = "tcp,cuda"
        print('os.environ[\"UCX_TLS\"] = {}'.format(os.environ["UCX_TLS"]))

    def tearDown(self) -> None:
        super().tearDown()
        if not self._has_master_port:
            del os.environ["MASTER_PORT"]
        if not self._has_ucx_tls:
            del os.environ["UCX_TLS"]

    @property
    def init_method(self):
        return "tcp://localhost:" + os.environ["MASTER_PORT"]
