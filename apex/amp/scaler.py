import torch
import logging

# from apex_C import scale_check_overflow

def scale_check_overflow_python(model_grad, scale, master_grad):
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
        if master_grad is not model_grad:
            master_grad.copy_(model_grad)
        master_grad.mul_(scale)
        return False
      
class LossScaler(object):
    warned_no_fused_kernel = False
    warned_fp16_grad = False
    has_fused_kernel = False

    def __init__(self,
                 loss_scale,
                 init_scale=2.**16,
                 scale_factor=2.,
                 scale_window=2000):
        if loss_scale == "dynamic":
            self.dynamic = True
            self._loss_scale = init_scale
        else:
            self.dynamic = False
            self._loss_scale = loss_scale
        self._max_loss_scale = 2.**24
        self._scale_seq_len = scale_window
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

    def unscale_and_update(self, model_params, master_params, scale):
        if LossScaler.has_fused_kernel:
            self._overflow_buf.zero_()
        self._has_overflow = False
        for model, master in zip(model_params, master_params):
            if model.grad is not None:
                if LossScaler.has_fused_kernel and master.grad.data.type() == "torch.cuda.FloatTensor":
                    LossScaler.scale_check_overflow_cuda(model.grad.data,
                                                         1./scale,
                                                         self._overflow_buf,
                                                         master.grad.data)
                else:
                    if (master.grad.data.type() != "torch.cuda.FloatTensor"
                            and not LossScaler.warned_fp16_grad):
                        logger = logging.getLogger("apex.amp")
                        logger.warning(
                            "Attempting to downscale {} grads. ".format(master.grad.data.type()) +
                            "Downscaling non-fp32 grads may indicate an error. "
                            "When using Amp, you don't need to call .half() on your model.")
                        LossScaler.warned_fp16_grad = True
                    self._has_overflow = scale_check_overflow_python(model.grad.data,
                                                                     1./scale,
                                                                     master.grad.data)
                    if self._has_overflow and self.dynamic:
                        break

        # If the fused kernel is available, we only need one D2H memcopy and sync.
        if LossScaler.has_fused_kernel and self.dynamic and not self._has_overflow:
            self._has_overflow = self._overflow_buf.item()

        if self._has_overflow and self.dynamic:
            should_skip = True
            self._loss_scale /= 2.
            self._unskipped = 0
        else:
            should_skip = False
            self._unskipped += 1

        if self._unskipped == self._scale_seq_len and self.dynamic:
            self._loss_scale = min(self._max_loss_scale, self._loss_scale * 2.)
            self._unskipped = 0

        return should_skip

def iter_params(param_groups):
    for group in param_groups:
        for p in group['params']:
            yield p
