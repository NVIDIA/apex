from typing import Tuple
import os
import subprocess
import sys
import unittest


# TODO(crcrpar): should move this to `apex._testing` or whatever in the near future.
RUN_SLOW_TESTS = os.getenv('APEX_RUN_WITH_SLOW_TESTS', '0') == '1'
SEVERALGPU_TEST = [
    "bert_minimal_test",
    "gpt_minimal_test",
    "dynamic_batchsize_test",
]


def get_multigpu_launch_option(min_gpu):
    should_skip = False
    import torch

    num_devices = torch.cuda.device_count()
    if num_devices < min_gpu:
        should_skip = True
    distributed_run_options = f"-m torch.distributed.run --nproc_per_node={num_devices}"
    return should_skip, distributed_run_options


def get_launch_option(test_filename) -> Tuple[bool, str]:
    should_skip = False
    for severalgpu_test in SEVERALGPU_TEST:
        if severalgpu_test in test_filename:
            return get_multigpu_launch_option(3)
    return should_skip, ""


def get_test_command(test_file: str) -> str:
    python_executable_path = sys.executable
    should_skip, launch_option = get_launch_option(test_file)
    if should_skip:
        return ""
    test_run_cmd = (
        f"{python_executable_path} {launch_option} {test_file} "
        "--micro-batch-size 2 --num-layers 16 --hidden-size 256 --num-attention-heads 8 --max-position-embeddings "
        "512 --seq-length 512 --global-batch-size 128"
    )
    if "bert" in test_file or "gpt" in test_file:
        import torch

        num_devices = torch.cuda.device_count()
        if "bert" in test_file:
            # "bert" uses the interleaving.
            tensor_model_parallel_size = 2 if num_devices % 2 == 0 and num_devices > 4 else 1
        if "gpt" in test_file:
            # "gpt" uses the non-interleaving.
            tensor_model_parallel_size = 2 if num_devices % 2 == 0 and num_devices >= 4 else 1
        pipeline_model_parallel_size = num_devices // tensor_model_parallel_size
        test_run_cmd += f" --pipeline-model-parallel-size {pipeline_model_parallel_size} --tensor-model-parallel-size {tensor_model_parallel_size}"

        if "bert" in test_file:
            test_run_cmd += f" --bert-no-binary-head"
    else:
        test_run_cmd += f" --use-cpu-initialization"
    return test_run_cmd


def _get_test_file(key):
    directory = os.path.dirname(__file__)
    test_file = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith("run_") and os.path.isfile(os.path.join(directory, f)) and key in f
    ][0]
    return test_file


@unittest.skipUnless(RUN_SLOW_TESTS, "this test takes long.")
class TestTransformer(unittest.TestCase):
    def _test_impl(self, key: str):
        test_file = _get_test_file(key)
        command = get_test_command(test_file)
        if command:
            subprocess.run(command, shell=True, check=True)
        else:
            self.skipTest("Appropriate command is not generated")

    def test_standalone_bert(self):
        self._test_impl("bert")

    def test_standalone_gpt(self):
        self._test_impl("gpt")

    def test_dynamic_batch_size(self):
        self._test_impl("dynamic_batchsize")


if __name__ == "__main__":
    unittest.main()
