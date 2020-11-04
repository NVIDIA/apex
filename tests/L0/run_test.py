import unittest
import sys

from apex.testing.common_utils import TEST_WITH_ROCM, skipIfRocm

test_dirs = ["run_amp", "run_fp16util", "run_optimizers", "run_fused_layer_norm", "run_pyprof_nvtx", "run_pyprof_data", "run_mlp"]

ROCM_BLACKLIST = [
    'run_pyprof_nvtx',
    'run_pyprof_data',
]

runner = unittest.TextTestRunner(verbosity=2)

errcode = 0

for test_dir in test_dirs:
    if (test_dir in ROCM_BLACKLIST) and TEST_WITH_ROCM:
        continue
    suite = unittest.TestLoader().discover(test_dir)

    print("\nExecuting tests from " + test_dir)

    result = runner.run(suite)

    if not result.wasSuccessful():
        errcode = 1

sys.exit(errcode)
