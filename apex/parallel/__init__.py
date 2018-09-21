from .distributed import DistributedDataParallel, Reducer
try:
    import syncbn
    print("using fused syncBN")
    from .optimized_sync_batchnorm import SyncBatchNorm
    from .optimized_sync_batchnorm import replace_with_SYNCBN
except:
    print("using non-fused syncBN, try install apex with 'python setup.py install --cuda_ext' to enable fused syncBN for better performance")
    from .sync_batchnorm import SyncBatchNorm
    from .sync_batchnorm import replace_with_SYNCBN
