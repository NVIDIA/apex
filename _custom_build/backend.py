import os
import subprocess
import sys
import sysconfig

from setuptools import build_meta as _orig
from setuptools.build_meta import *  # noqa: F401


def create_symlink_to_installed(names=("torch", "torch.egg-link")):
    paths = sysconfig.get_paths()
    auxiliary_paths_for_torch = list({paths['purelib'], paths['platlib']})
    path_to_installed_torch = ""
    for directory in auxiliary_paths_for_torch:
        for name in names:
            tmp_torch_path = os.path.join(directory, name)
            if os.path.exists(tmp_torch_path):
                path_to_installed_torch = tmp_torch_path
                break
    if not path_to_installed_torch:
        raise RuntimeError("`torch` is required but not installed")

    symlink_dst = ""
    for d in sys.path:
        if "site-packages" in d:
            symlink_dst = d
            break
        if "dist-packages" in d:
            symlink_dst = d
    if not symlink_dst:
        raise RuntimeError("no appropriate site-packages or dist-packages found")
    command = f"cp -r {path_to_installed_torch} {symlink_dst}"
    subprocess.check_call(command.split(" "))


def get_requires_for_build_wheel(config_settings=None):
    create_symlink_to_installed(("torch", "torch.egg-link"))
    create_symlink_to_installed(("torchgen",))
    return _orig.get_requires_for_build_wheel(config_settings)
