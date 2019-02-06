import torch
import logging

# from apex_C import scale_check_overflow

def scale_check_overflow_python(d_grads, scale):
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
    warned_no_fused_kernel = False
    warned_fp16_grad = False
    has_fused_kernel = False

    def __init__(self):
        self._loss_scale = 2.**16
        self._max_loss_scale = 2.**24
        self._scale_seq_len = 2000
        self._unskipped = 0
        self._has_overflow = False
        try:
            import amp_C
            LossScaler.has_fused_kernel = True
            LossScaler.scale_check_overflow_cuda = amp_C.scale_check_overflow
            self._overflow_buf = torch.cuda.IntTensor([0])
        except ImportError as err:
            if not LossScaler.warned_no_fused_kernel:
                print("Warning:  Amp fused downscale kernel is unavailable, possibly because apex "
                      "was installed without --cuda_ext.  Using Python fallback.  ImportError was: ",
                      err)
            LossScaler.has_fused_kernel = False
            LossScaler.warned_no_fused_kernel = True

    def loss_scale(self):
        return self._loss_scale

    def unscale_and_update(self, param_groups, scale):
        if LossScaler.has_fused_kernel:
            self._overflow_buf.zero_()
        self._has_overflow = False
        for p in iter_params(param_groups):
            if p.grad is not None:
                if LossScaler.has_fused_kernel and p.grad.data.type() == "torch.cuda.FloatTensor":
                    LossScaler.scale_check_overflow_cuda(p.grad.data,
                                                         1./scale,
                                                         self._overflow_buf,
                                                         p.grad.data)
                else:
                    if (p.grad.data.type() != "torch.cuda.FloatTensor"
                            and not LossScaler.warned_fp16_grad):
                        logger = logging.getLogger("apex.amp")
                        logger.warning("Incoming grads are not fp32 (not master grads). "
                                       "Downscaling non-fp32 grads may indicate an error. "
                                       "When using Amp, you don't need to call .half() on your model.")
                        LossScaler.warned_fp16_grad = True
                    self._has_overflow = scale_check_overflow_python(p.grad.data,
                                                                     1./scale)
                    if self._has_overflow:
                        break

        # If the fused kernel is available, we only need one D2H memcopy and sync.
        if LossScaler.has_fused_kernel and not self._has_overflow:
            self._has_overflow = self._overflow_buf.item()

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
