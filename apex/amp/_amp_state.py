# This is a "header object" that allows different amp modules to communicate.
# I'm a C++ guy, not a python guy.  I decided this approach because it seemed most C++-like.
# But apparently it's ok:
# http://effbot.org/pyfaq/how-do-i-share-global-variables-across-modules.htm
import os
import torch

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 0:
    import collections.abc as container_abcs
else:
    from torch._six import container_abcs


class AmpState(object):
    def __init__(self):
        self.hard_override=False
        self.allow_incoming_model_not_fp32 = False
        self.verbosity=1


# Attribute stash.  Could also just stash things as global module attributes.
_amp_state = AmpState()


def warn_or_err(msg):
    if _amp_state.hard_override:
        print("Warning:  " + msg)
    else:
        raise RuntimeError(msg)
        # I'm not sure if allowing hard_override is a good idea.
        # + "  If you're sure you know what you're doing, supply " +
        #                    "hard_override=True to amp.initialize.")


def maybe_print(msg, rank0=False):
    distributed = torch.distributed.is_available() and \
        torch.distributed.is_initialized() and \
        torch.distributed.get_world_size() > 1
    if _amp_state.verbosity > 0:
        if rank0:
            if distributed:
                if torch.distributed.get_rank() == 0:
                    print(msg)
            else:
                print(msg)
        else:
            print(msg)


# def iter_params(param_groups):
#     for group in param_groups:
#         for p in group['params']:
#             yield p


def master_params(optimizer):
    """
    Generator expression that iterates over the params owned by ``optimizer``.

    Args:
        optimizer: An optimizer previously returned from ``amp.initialize``.
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            yield p
