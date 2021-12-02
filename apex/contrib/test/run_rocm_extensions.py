import unittest
import sys


test_dirs = ["groupbn", "layer_norm", "multihead_attn", "."] # "." for test_label_smoothing.py
ROCM_BLACKLIST = [
    "groupbn",
    "layer_norm"
]

runner = unittest.TextTestRunner(verbosity=2)

errcode = 0

for test_dir in test_dirs:
    if test_dir in ROCM_BLACKLIST:
        continue
    suite = unittest.TestLoader().discover(test_dir)

    print("\nExecuting tests from " + test_dir)

    result = runner.run(suite)

    if not result.wasSuccessful():
        errcode = 1

sys.exit(errcode)
