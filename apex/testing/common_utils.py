'''
This file contains common utility functions for running the unit tests on ROCM.
'''

import torch
import os
import sys
from functools import wraps
import unittest


TEST_WITH_ROCM = os.getenv('APEX_TEST_WITH_ROCM', '0') == '1'

## Wrapper to skip the unit tests.
def skipIfRocm(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if TEST_WITH_ROCM:
            raise unittest.SkipTest("test doesn't currently work on ROCm stack.")
        else:
            fn(*args, **kwargs)
    return wrapper
