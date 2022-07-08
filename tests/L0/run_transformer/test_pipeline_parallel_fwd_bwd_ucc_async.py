import os
import logging

import torch
from torch.testing._internal import common_utils
from torch.testing._internal import common_distributed

from apex.transformer.testing.distributed_test_base import UccDistributedTestBase
from apex.transformer.testing.distributed_test_base import HAS_TORCH_UCC
from apex.transformer.testing.distributed_test_base import HAS_TORCH_UCC_COMPAT_NVIDIA_DRIVER

from test_pipeline_parallel_fwd_bwd import PipelineParallelForwardBackwardTestBase

logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("apex").setLevel(logging.WARNING)

weight_coeff = 1024

class UccPipelineParallelForwardBackwardUnderLoadTest(UccDistributedTestBase, PipelineParallelForwardBackwardTestBase):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.GLOBAL_BATCH_SIZE = 8192
        self.MICRO_BATCH_SIZE =  self.GLOBAL_BATCH_SIZE // self.world_size
        self.HIDDEN_SIZE = 256
        self.NUM_EPOCHS = 1
        self.deallocate_options = (False,)
        self.dtypes = (torch.float32,)

        if "NUM_EPOCHS" in os.environ:
            self.NUM_EPOCHS = int(os.environ["NUM_EPOCHS"])

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), 8)

if __name__ == "__main__":
    common_distributed.TIMEOUT_DEFAULT = 500
    common_utils.run_tests()
