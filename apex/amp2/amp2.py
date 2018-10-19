from apex import amp

# TODO: what we want is for this to construct a class object that exposes (roughly)
# the same API as this module-level API (thereby ensuring there's only one of them,
# since modules are imported only once per interpreter).
# Basically, not this.
_amp_handle = amp.handle.NoOpHandle()

# TODO: everything about this is a hack right now. We should re-work the API more carefully.
def enable_automatic_conversion():
    global _amp_handle
    if _amp_handle.is_active():
        raise RuntimeError('Cannot call `enable_automatic_conversion` more than once.')
    _amp_handle = amp.init(enabled=True)

def clear_amp_cache():
    global _amp_handle
    if _amp_handle.has_cache:
        _amp_handle.cache.clear()
