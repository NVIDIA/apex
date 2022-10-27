"""L0 Tests Runner.

How to run this script?

1. Run all the tests: `python /path/to/apex/tests/L0/run_test.py` If you want an xml report,
    pass `--xml-report`, i.e. `python /path/to/apex/tests/L0/run_test.py --xml-report` and
    the file is created in `/path/to/apex/tests/L0`.
2. Run one of the tests (e.g. fused layer norm):
    `python /path/to/apex/tests/L0/run_test.py --include run_fused_layer_norm`
3. Run two or more of the tests (e.g. optimizers and fused layer norm):
    `python /path/to/apex/tests/L0/run_test.py --include run_optimizers run_fused_layer_norm`
"""
import argparse
import os
import unittest
import sys


TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIRS = [
    "run_amp",
    "run_deprecated",
    "run_fp16util",
    "run_optimizers",
    "run_fused_layer_norm",
    "run_mlp",
    "run_transformer",
    "run_instance_norm_nvfuser",
]
DEFAULT_TEST_DIRS = [
    "run_optimizers",
    "run_fused_layer_norm",
    "run_mlp",
    "run_transformer",
    "run_instance_norm_nvfuser",
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
    parser.add_argument(
        "--xml-report",
        action="store_true",
        help="pass this argument to get a junit xml report. (requires `xmlrunner`)",
    )
    args, _ = parser.parse_known_args()
    return args


def main(args: argparse.Namespace) -> None:
    test_runner_kwargs = {"verbosity": 2}
    Runner = unittest.TextTestRunner
    if args.xml_report:
        import xmlrunner
        from datetime import date  # NOQA
        Runner = xmlrunner.XMLTestRunner

    errcode = 0
    for test_dir in args.include:
        if args.xml_report:
            this_dir = os.path.abspath(os.path.dirname(__file__))
            xml_output = os.path.join(
                this_dir,
                f"""TEST_{test_dir}_{date.today().strftime("%y%m%d")}""",
            )
            test_runner_kwargs["output"] = xml_output

        runner = Runner(**test_runner_kwargs)
        test_dir = os.path.join(TEST_ROOT, test_dir)
        suite = unittest.TestLoader().discover(test_dir)

        print("\nExecuting tests from " + test_dir)

        result = runner.run(suite)

        if not result.wasSuccessful():
            errcode = 1

    sys.exit(errcode)


if __name__ == '__main__':
    args = parse_args()
    main(args)
