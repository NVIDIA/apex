"""L0 Tests Runner.

How to run this script?

1. Run all the tests: `python /path/to/apex/tests/L0/run_test.py`
2. Run one of the tests (e.g. fused layer norm):
    `python /path/to/apex/tests/L0/run_test.py --include run_fused_layer_norm`
3. Run two or more of the tests (e.g. optimizers and fused layer norm):
    `python /path/to/apex/tests/L0/run_test.py --include run_optimizers run_fused_layer_norm`
"""
import argparse
import os
import unittest
import sys

xml_output = None
test_runner_kwargs = {"verbosity": 2}
Runner = unittest.TextTestRunner
try:
    import xmlrunner
except ImportError:
    pass
else:
    if os.getenv("CI", "0") == "1":
        from datetime import date  # NOQA
        this_dir = os.path.abspath(os.path.dirname(__file__))
        xml_filename = os.path.join(
            this_dir,
            f"""{date.today().strftime("%y%m%d")}.xml""",
        )
        print(f"\n\tReport file: {xml_filename}\n")
        xml_output = open(xml_filename, "w")
        test_runner_kwargs["output"] = xml_output
        Runner = xmlrunner.XMLTestRunner


TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIRS = [
    "run_amp",
    "run_fp16util",
    "run_optimizers",
    "run_fused_layer_norm",
    "run_mlp",
    "run_transformer",
]
DEFAULT_TEST_DIRS = [
    "run_optimizers",
    "run_fused_layer_norm",
    "run_mlp",
    "run_transformer",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="L0 test runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=TEST_DIRS,
        default=DEFAULT_TEST_DIRS,
        help="select a set of tests to run (defaults to ALL tests).",
    )
    args, _ = parser.parse_known_args()
    return args


def main(args: argparse.Namespace) -> None:
    runner = Runner(**test_runner_kwargs)
    errcode = 0
    for test_dir in args.include:
        test_dir = os.path.join(TEST_ROOT, test_dir)
        print(test_dir)
        suite = unittest.TestLoader().discover(test_dir)

        print("\nExecuting tests from " + test_dir)

        result = runner.run(suite)

        if not result.wasSuccessful():
            errcode = 1

    if xml_output is not None:
        xml_output.close()

    sys.exit(errcode)


if __name__ == '__main__':
    args = parse_args()
    main(args)
