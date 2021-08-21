import os
import subprocess
import sys
import unittest


def run_mpu_tests():
    python_executable_path = sys.executable
    repository_root = subprocess.check_output(
        "git rev-parse --show-toplevel").decode(sys.stdout.encoding).strip()
    directory = os.path.abspath(os.path.join(repository_root, "tests/mpu"))
    files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.startswith("test_") and os.path.isfile(os.path.join(directory, f))
    ]
    print("#######################################################")
    print(f"# Python executable path: {python_executable_path}")
    print(f"# {len(files)} tests: {files}")
    print("#######################################################")
    errors = []
    for i, test_file in enumerate(files, 1):
        test_run_cmd = f"NVIDIA_TF32_OVERRIDE=0  {python_executable_path} {test_file} --micro-batch-size 2 --num-layers 1 --hidden-size 256 --num-attention-heads 8 --max-position-embeddings 32 --encoder-seq-length 32 --use-cpu-initialization"  # NOQA
        print(f"### {i} / {len(files)}: cmd: {test_run_cmd}")
        try:
            output = subprocess.check_output(
                test_run_cmd, shell=True
            ).decode(sys.stdout.encoding).strip()
        except Exception as e:
            errors.append((test_file, str(e)))
        else:
            if '>> passed the test :-)' not in output:
                errors.append(test_file, output)
    else:
        if not errors:
            print("### PASSED")
        else:
            print("### FAILED")
            print(f"{len(errors)} out of {len(files)} tests failed")
            for (filename, log) in errors:
                print(f"File: {filename}\nLog: {log}")
            raise RuntimeError(f"{len(errors)} out of {len(files)} tests failed")


class TestMPU(unittest.TestCase):

    def test_mpu(self):
        run_mpu_tests()


if __name__ == '__main__':
    unittest.main()
