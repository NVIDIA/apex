from _apex_gpu_direct_storage import _GDSFile
from contextlib import contextmanager

@contextmanager
def GDSFile(filename, mode):
    assert type(filename) == str
    assert type(mode) == str
    try:
        file_handle = _GDSFile(filename, mode)
        yield file_handle
    finally:
        file_handle.close()
        del file_handle
