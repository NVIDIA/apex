from _apex_gpu_direct_storage import _GDSFile
from contextlib import contextmanager


@contextmanager
def GDSFile(filename, mode):
    assert type(filename) == str
    assert type(mode) == str
    try:
        from apex import deprecated_warning

        deprecated_warning(
            "`gpu_direct_storage.GDSFile` is deprecated and will be removed in September 2025. "
            "We encourage you to use `torch.cuda.gds` module of PyTorch as a replacement. "
            "Its documentation is available at https://docs.pytorch.org/docs/stable/cuda.html#gpudirect-storage-prototype"
        )
        file_handle = _GDSFile(filename, mode)
        yield file_handle
    finally:
        file_handle.close()
        del file_handle
