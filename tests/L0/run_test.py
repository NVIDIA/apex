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
    parser.add_argument(
        "--xml-report",
        default=None,
        action="store_true",
        help="[deprecated] pass this argument to get a junit xml report. Use `--xml-dir`. (requires `xmlrunner`)",
    )
    parser.add_argument(
        "--xml-dir",
        default=None,
        type=str,
        help="Directory to save junit test reports. (requires `xmlrunner`)",
    )
    args, _ = parser.parse_known_args()
    return args


def main(args: argparse.Namespace) -> None:
    test_runner_kwargs = {"verbosity": 2}
    Runner = unittest.TextTestRunner

    xml_dir = None
    if (args.xml_report is not None) or (args.xml_dir is not None):
        if args.xml_report is not None:
            import warnings
            warnings.warn("The option of `--xml-report` is deprecated", FutureWarning)

        import xmlrunner
        from datetime import date  # NOQA
        Runner = xmlrunner.XMLTestRunner
        if args.xml_report:
            xml_dir = os.path.abspath(os.path.dirname(__file__))
        else:
            xml_dir = os.path.abspath(args.xml_dir)
        if not os.path.exists(xml_dir):
            os.makedirs(xml_dir)

    errcode = 0
    for test_dir in args.include:
        if xml_dir is not None:
            xml_output = os.path.join(
                xml_dir,
                f"""TEST_{test_dir}_{date.today().strftime("%y%m%d")}""",
            )
            if not os.path.exists(xml_output):
                os.makedirs(xml_output)
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
