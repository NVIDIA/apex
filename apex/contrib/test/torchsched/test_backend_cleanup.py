import subprocess
import sys
import unittest


# Importing torchsched also registers cuDNN-specific ops and changes torch.compile.
# Isolate those module-global effects in a child, and skip only the unrelated op
# registration so the real Python scheduler/GraphLowering can be tested on CPU.
_SETUP = """
from unittest.mock import patch
import torch
original_compile = torch.compile
with patch.object(torch.ops, "import_module"):
    from apex.contrib.torchsched.backend import enable_multi_stream_scheduling
from torch._inductor.graph import GraphLowering
original_codegen = GraphLowering.codegen
"""


class TestCompilerPatchCleanup(unittest.TestCase):
    def run_isolated(self, body):
        result = subprocess.run(
            [sys.executable, "-c", _SETUP + body],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_restores_codegen_after_exception(self):
        self.run_isolated("""
for error_type in (RuntimeError, ValueError, KeyboardInterrupt):
    error = error_type("controlled compiler failure")
    def fail():
        assert GraphLowering.codegen is not original_codegen
        raise error
    try:
        enable_multi_stream_scheduling(fail)()
    except BaseException as actual:
        assert actual is error
    else:
        raise AssertionError("compiler exception was swallowed")
    assert GraphLowering.codegen is original_codegen
""")

    def test_success_preserves_arguments_result_and_metadata(self):
        self.run_isolated("""
result = object()
def compile_fn(value, *, flag):
    assert value == 3 and flag is True
    assert GraphLowering.codegen is not original_codegen
    return result
wrapped = enable_multi_stream_scheduling(compile_fn)
assert wrapped.__wrapped__ is compile_fn
assert wrapped.__name__ == compile_fn.__name__
assert wrapped(3, flag=True) is result
assert GraphLowering.codegen is original_codegen
""")

    def test_standard_cpu_compile_works_after_failure(self):
        self.run_isolated("""
def fail():
    raise RuntimeError("controlled compiler failure")
try:
    enable_multi_stream_scheduling(fail)()
except RuntimeError:
    pass
assert GraphLowering.codegen is original_codegen
# Use the original entry point, not torchsched's import-time default wrapper.
fn = original_compile(lambda x: x.sin() + x.square(), backend="inductor", fullgraph=True)
x = torch.linspace(-1, 1, 8, requires_grad=True)
y = fn(x)
torch.testing.assert_close(y, x.sin() + x.square())
y.sum().backward()
torch.testing.assert_close(x.grad, x.detach().cos() + 2 * x.detach())
assert GraphLowering.codegen is original_codegen
""")


if __name__ == "__main__":
    unittest.main()
