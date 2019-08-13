import unittest
import sys

test_dirs = ["run_amp", "run_fp16util", "run_optimizers", "run_fused_layer_norm", "run_pyprof_nvtx"]

runner = unittest.TextTestRunner(verbosity=2)

errcode = 0

for test_dir in test_dirs:
    suite = unittest.TestLoader().discover(test_dir)

    print("\nExecuting tests from " + test_dir)

    result = runner.run(suite)

    if not result.wasSuccessful():
        errcode = 1

sys.exit(errcode)
