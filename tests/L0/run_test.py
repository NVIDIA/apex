import os
import unittest
import sys


TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIRS = [
    "run_amp",
    "run_fp16util",
    "run_optimizers",
    "run_fused_layer_norm",
    "run_pyprof_nvtx",
    "run_pyprof_data",
    "run_mlp",
    "run_mpu",
]


def main():
    runner = unittest.TextTestRunner(verbosity=2)
    errcode = 0
    for test_dir in TEST_DIRS:
        test_dir = os.path.join(TEST_ROOT, test_dir)
        print(test_dir)
        suite = unittest.TestLoader().discover(test_dir)

        print("\nExecuting tests from " + test_dir)

        result = runner.run(suite)

        if not result.wasSuccessful():
            errcode = 1

    sys.exit(errcode)


if __name__ == '__main__':
    main()
