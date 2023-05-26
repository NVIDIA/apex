import logging

import torch.testing
from torch.testing._internal import common_utils

logging.getLogger("torch").setLevel(logging.WARNING)

from apex.transformer import parallel_state
from apex.transformer.tensor_parallel import data as data_utils
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase
from apex.transformer.testing.distributed_test_base import UccDistributedTestBase

logging.getLogger("torch").setLevel(logging.WARNING)


class BroadcastDataTestBase:
    def test_broadcast_data(self):
        tensor_model_parallel_world_size: int = self.world_size // (
            1 + self.world_size > 1
        )
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=tensor_model_parallel_world_size
        )

        target_key_size = {
            "key1": [7, 11],
            "key2": [8, 2, 1],
            "key3": [13],
            "key4": [5, 1, 2],
            "key5": [5, 12],
        }
        keys = [k for k in target_key_size]

        data = {}
        data_t = {}
        with torch.no_grad():
            for key in target_key_size:
                data[key] = torch.randint(0, 1000, size=target_key_size[key])
                data_t[key] = data[key].clone()
            # "key_x" is supposed to be ignored.
            data["key_x"] = torch.rand(5)
            data_t["key_x"] = data["key_x"].clone()
        if parallel_state.get_tensor_model_parallel_rank() != 0:
            data = None

        data_utils._check_data_types(keys, data_t, torch.int64)
        key_size, _, _ = data_utils._build_key_size_numel_dictionaries(keys, data)

        for key in keys:
            self.assertEqual(target_key_size[key], key_size[key])

        broadcasted_data = data_utils.broadcast_data(keys, data, torch.int64)
        for key in keys:
            self.assertEqual(broadcasted_data[key], data_t[key].cuda())

        parallel_state.destroy_model_parallel()


class NcclBroadcastDataTest(BroadcastDataTestBase, NcclDistributedTestBase): pass
class UccBroadcastDataTest(BroadcastDataTestBase, UccDistributedTestBase): pass


if __name__ == "__main__":
    common_utils.run_tests()
