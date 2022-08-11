import logging
from typing import List, Optional

from torch.testing._internal import common_utils

logging.getLogger("torch").setLevel(logging.WARNING)

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel.utils import (
    _reconfigure_microbatch_calculator,
    get_micro_batch_size,
    get_num_microbatches,
    get_current_global_batch_size,
    update_num_microbatches,
)
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase
from apex.transformer.testing.distributed_test_base import UccDistributedTestBase

logging.getLogger("apex").setLevel(logging.WARNING)


class MicrobatchCalculatorTestBase:

    GLOBAL_BATCH_SIZE: int = 1024
    MICRO_BATCH_SIZE: int = 1

    def _test(self, rampup_batch_size: Optional[List[int]]) -> None:
        for data_parallel_size in range(1, self.world_size + 1):

            expected_global_batch_size = self.GLOBAL_BATCH_SIZE
            expected_micro_batch_size = self.MICRO_BATCH_SIZE
            if rampup_batch_size:
                expected_global_batch_size = rampup_batch_size[0]
                num_consumed_samples = 0
                step_of_global_batch_size = rampup_batch_size[1]
                threshold = rampup_batch_size[2]

            if data_parallel_size > 1 and data_parallel_size % 2 != 0:
                continue
            if self.world_size % data_parallel_size != 0:
                continue
            msg = f"data_parallel_size: {data_parallel_size}"
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size_=self.world_size // data_parallel_size,
                pipeline_model_parallel_size_=1,
            )
            self.assertEqual(data_parallel_size, parallel_state.get_data_parallel_world_size(), msg=msg)

            _reconfigure_microbatch_calculator(
                self.rank,
                rampup_batch_size,
                self.GLOBAL_BATCH_SIZE,
                self.MICRO_BATCH_SIZE,
                data_parallel_size,
            )

            self.assertEqual(get_micro_batch_size(), expected_micro_batch_size, msg=msg)
            self.assertEqual(get_num_microbatches(), expected_global_batch_size / expected_micro_batch_size / data_parallel_size, msg=msg)
            current_global_batch_size = get_current_global_batch_size()
            self.assertEqual(current_global_batch_size, expected_global_batch_size, msg=msg)

            # Make sure `global_batch_size` equals to the final global batch size after
            # certain number of updates.
            if rampup_batch_size:
                update_num_microbatches(current_global_batch_size)
                for i in range(100):
                    current_global_batch_size = get_current_global_batch_size()
                    update_num_microbatches(current_global_batch_size)
                current_global_batch_size = get_current_global_batch_size()
                self.assertEqual(get_current_global_batch_size(), self.GLOBAL_BATCH_SIZE, msg=msg)
            parallel_state.destroy_model_parallel()

    def test_constant_microbatch_calculator(self):
        self._test(rampup_batch_size=None)

    def test_dynamic_microbatch_calculator(self):
        self._test(rampup_batch_size=[256, 128, 500])


class NcclMicrobatchCalculatorTest(MicrobatchCalculatorTestBase, NcclDistributedTestBase): pass
class UccMicrobatchCalculatorTest(MicrobatchCalculatorTestBase, UccDistributedTestBase): pass


if __name__ == "__main__":
    common_utils.run_tests()
