# Inspired by https://github.com/python-pillow/Pillow/pull/7171.

import sys

from setuptools.build_meta import *  # noqa: F401, F403

backend_class = build_wheel.__self__.__class__


class _CustomBuildMetaBackend(backend_class):
    def run_setup(self, setup_script="setup.py"):
        if self.config_settings:

            def config_has(key, value):
                settings = self.config_settings.get(key)
                if settings:
                    if not isinstance(settings, list):
                        settings = [settings]
                    return value in settings

            flags = []
            for ext in (
                "cpp_ext",
                "cuda_ext",
                "distributed_adam",
                "distributed_lamb",
                "permutation_search",
                "bnp",
                "xentropy",
                "focal_loss",
                "index_mul_2d",
                "deprecated_fused_adam",
                "deprecated_fused_lamb",
                "fast_layer_norm",
                "fmha",
                "fast_multihead_attn",
                "transducer",
                "cudnn_gbn",
                "peer_memory",
                "nccl_p2p",
                "fast_bottleneck",
                "fused_conv_bias_relu",
            ):
                if ext not in self.config_settings:
                    continue

                if config_has(ext, "enable"):
                    flags.append("--" + ext)
                elif not config_has(ext, "disable"):
                    raise ValueError(
                        f'unknown argument value for {ext}; must be either '
                        f'"enable" or "disable"'
                    )

            if flags:
                sys.argv = sys.argv[:1] + ["build_ext"] + flags + sys.argv[1:]
        return super().run_setup(setup_script)

    def build_wheel(
        self, wheel_directory, config_settings=None, metadata_directory=None
    ):
        self.config_settings = config_settings
        return super().build_wheel(wheel_directory, config_settings, metadata_directory)


build_wheel = _CustomBuildMetaBackend().build_wheel
