"""Regression tests for the APEX_ASP_CACHE_DIR path-traversal fix (CWE-22/CWE-73).

These exercise apex/contrib/sparsity/permutation_search_kernels/exhaustive_search.py
directly. They do not require CUDA: the permutation generation falls back to a pure
CPU/numpy path when the search kernels are not built.
"""

import importlib
import os
import sys
import types
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _load_exhaustive_search():
    # Importing apex.contrib.sparsity normally pulls in torchvision via asp.py, which
    # is unrelated to the cache-path code under test. Register lightweight namespace
    # packages so the kernels subpackage's relative imports resolve without it.
    for name, rel in [
        ("apex", "apex"),
        ("apex.contrib", "apex/contrib"),
        ("apex.contrib.sparsity", "apex/contrib/sparsity"),
    ]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [os.path.join(REPO_ROOT, rel)]
            sys.modules[name] = mod
    return importlib.import_module(
        "apex.contrib.sparsity.permutation_search_kernels.exhaustive_search"
    )


class TestAspCachePath(unittest.TestCase):
    def setUp(self):
        self.mod = _load_exhaustive_search()
        self._prev_env = os.environ.get(self.mod.ASP_CACHE_DIR_ENV_VAR)
        self._prev_cwd = os.getcwd()
        self._tmp = self._mkdtemp()
        # Run from a temp dir so the default ".cache" base lives there.
        os.chdir(self._tmp)
        # Clear the module-level memoization so each test actually writes.
        self.mod.unique_permutation_list = {}

    def tearDown(self):
        os.chdir(self._prev_cwd)
        if self._prev_env is None:
            os.environ.pop(self.mod.ASP_CACHE_DIR_ENV_VAR, None)
        else:
            os.environ[self.mod.ASP_CACHE_DIR_ENV_VAR] = self._prev_env

    def _mkdtemp(self):
        import tempfile

        return os.path.realpath(tempfile.mkdtemp())

    def test_normal_path_within_base_is_used(self):
        # A cache dir nested under the allowed default base is honored.
        allowed_base = os.path.realpath(self.mod.ASP_CACHE_DIR_DEFAULT)
        nested = os.path.join(allowed_base, "sub")
        os.environ[self.mod.ASP_CACHE_DIR_ENV_VAR] = nested

        result = self.mod.generate_all_unique_combinations(4, 4)

        self.assertTrue(len(result) >= 1)
        expected_file = os.path.join(nested, "permutations_4_4.npy")
        self.assertTrue(os.path.exists(expected_file), f"missing {expected_file}")

    def test_traversal_attempt_is_rejected(self):
        # An attacker-controlled value pointing outside the allowed base must not be
        # used as the write destination; the code falls back to the safe default.
        escape_dir = self._mkdtemp()  # an arbitrary writable dir outside ".cache"
        # ".." traversal that resolves to a fresh, unique location outside the base.
        evil_target = os.path.realpath(os.path.join(escape_dir, "..", os.path.basename(escape_dir) + "_evil"))
        os.environ[self.mod.ASP_CACHE_DIR_ENV_VAR] = os.path.join(escape_dir, "..", os.path.basename(escape_dir) + "_evil")

        result = self.mod.generate_all_unique_combinations(4, 4)

        self.assertTrue(len(result) >= 1)
        # Nothing was written to the attacker-controlled location.
        leaked = os.path.join(evil_target, "permutations_4_4.npy")
        self.assertFalse(os.path.exists(leaked), f"write escaped to {leaked}")
        # The file landed in the safe default base instead.
        safe_file = os.path.join(
            os.path.realpath(self.mod.ASP_CACHE_DIR_DEFAULT), "permutations_4_4.npy"
        )
        self.assertTrue(os.path.exists(safe_file), f"missing safe write {safe_file}")

    def test_resolve_cache_dir_helper(self):
        # Direct unit check on the resolver.
        allowed_base = os.path.realpath(self.mod.ASP_CACHE_DIR_DEFAULT)

        os.environ.pop(self.mod.ASP_CACHE_DIR_ENV_VAR, None)
        self.assertEqual(self.mod._resolve_cache_dir(), allowed_base)

        os.environ[self.mod.ASP_CACHE_DIR_ENV_VAR] = "/etc"
        self.assertEqual(self.mod._resolve_cache_dir(), allowed_base)

        within = os.path.join(allowed_base, "ok")
        os.environ[self.mod.ASP_CACHE_DIR_ENV_VAR] = within
        self.assertEqual(self.mod._resolve_cache_dir(), within)


if __name__ == "__main__":
    unittest.main()
