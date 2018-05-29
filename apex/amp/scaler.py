import torch

from ._C import scale_lib

class LossScaler(object):
    def __init__(self):
        self._loss_scale = 2.**16
        self._max_loss_scale = 2.**24
        self._scale_seq_len = 2000
        self._unskipped = 0
        self._overflow_buf = torch.cuda.ByteTensor(1024,)

    def loss_scale(self):
        return self._loss_scale

    def unscale_and_update(self, param_groups, scale):
        self._overflow_buf.zero_()
        for p in iter_params(param_groups):
            if p.grad is not None:
                scale_lib.scale_check_overflow(p.grad.data,
                                               1. / scale,
                                               self._overflow_buf)

        if self._overflow_buf.any():
            should_skip = True
            self._loss_scale /= 2.
            self._unskipped = 0
        else:
            should_skip = False
            self._unskipped += 1

        if self._unskipped == self._scale_seq_len:
            self._loss_scale = min(self._max_loss_scale, self._loss_scale * 2.)
            self._unskipped = 0

        return should_skip

def iter_params(param_groups):
    for group in param_groups:
        for p in group['params']:
            yield p
