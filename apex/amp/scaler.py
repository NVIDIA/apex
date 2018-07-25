import torch

# from apex_C import scale_check_overflow

# Python stopgap, until we get a future-proof kernel into upstream
def scale_check_overflow(d_grads, scale):
    # Exception handling for 18.04 compatibility
    try:
        cpu_sum = float(d_grads.float().sum())
    except RuntimeError as instance:
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
            return True
        d_grads.mul_(scale)
        return False
      
class LossScaler(object):
    def __init__(self):
        self._loss_scale = 2.**16
        self._max_loss_scale = 2.**24
        self._scale_seq_len = 2000
        self._unskipped = 0
        self._has_overflow = False
        # self._overflow_buf = torch.cuda.ByteTensor(1024,)

    def loss_scale(self):
        return self._loss_scale

    def unscale_and_update(self, param_groups, scale):
        # self._overflow_buf.zero_()
        self._has_overflow = False
        for p in iter_params(param_groups):
            if p.grad is not None:
                self._has_overflow = scale_check_overflow(p.grad.data,
                                                          1. / scale)
            if self._has_overflow:  
                break

        # if self._overflow_buf.any():
        if self._has_overflow:
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
