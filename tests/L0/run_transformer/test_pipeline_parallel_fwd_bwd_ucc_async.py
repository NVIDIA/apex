import os
import logging
import itertools
import re
from typing import Optional, Tuple, List
import unittest

import torch
from torch.testing._internal import common_utils
from torch.testing._internal import common_cuda
from torch.testing._internal import common_distributed

from apex._autocast_utils import _get_autocast_dtypes
from apex.transformer import parallel_state
from apex.transformer.enums import ModelType
from apex.transformer.pipeline_parallel import utils as pp_utils
from apex.transformer.pipeline_parallel.schedules.common import (
    FwdStepFunc,
    build_model,
    _get_params_for_weight_decay_optimization,
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import (
    forward_backward_no_pipelining,
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving import (
    _forward_backward_pipelining_with_interleaving,
)
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
    forward_backward_pipelining_without_interleaving,
)
from apex.transformer.testing.distributed_test_base import UccDistributedTestBase
from apex.transformer.testing.distributed_test_base import HAS_TORCH_UCC
from apex.transformer.testing.distributed_test_base import HAS_TORCH_UCC_COMPAT_NVIDIA_DRIVER
from apex.transformer.testing import commons as testing_utils

from test_pipeline_parallel_fwd_bwd import PipelineParallelForwardBackwardTestBase
from test_pipeline_parallel_fwd_bwd import get_target_loss_and_model

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
