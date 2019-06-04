import torch
from ..multi_tensor_apply import multi_tensor_applier
from ._amp_state import _amp_state, master_params, maybe_print
from itertools import product

def scale_check_overflow_python(model_grad, master_grad, scale, check_overflow=False):
    # Exception handling for 18.04 compatibility
    if check_overflow:
        cpu_sum = float(model_grad.float().sum())
        if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
            return True

    if master_grad is not model_grad: # copy_ probably internally short-circuits this
        master_grad.copy_(model_grad)
    if scale != 1.0:
        master_grad.mul_(scale)
    return False

def axpby_check_overflow_python(model_grad, stashed_grad, master_grad, scale, check_overflow=False):
    # Exception handling for 18.04 compatibility
    if check_overflow:
        cpu_sum = float(model_grad.float().sum())
        if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
            return True

    # if master_grad is not model_grad: # copy_ probably internally short-circuits this
    #     master_grad.copy_(model_grad)
    assert stashed_grad.dtype == master_grad.dtype
    converted_model_grad = model_grad.to(master_grad.dtype)
    stashed_grad.add_(scale, converted_model_grad)
    master_grad.data = stashed_grad.data
    return False

class LossScaler(object):
    warned_no_fused_kernel = False
    warned_unscaling_non_fp32_grad = False
    has_fused_kernel = False

    def __init__(self,
                 loss_scale,
                 init_scale=2.**16,
                 scale_factor=2.,
                 scale_window=2000,
                 min_loss_scale=None,
                 max_loss_scale=2.**24):
        if loss_scale == "dynamic":
            self.dynamic = True
            self._loss_scale = init_scale
        else:
            self.dynamic = False
            self._loss_scale = loss_scale
        self._max_loss_scale = max_loss_scale
        self._min_loss_scale = min_loss_scale
        self._scale_seq_len = scale_window
        self._unskipped = 0
        self._has_overflow = False
        self._overflow_buf = torch.cuda.IntTensor([0])
        if multi_tensor_applier.available:
            import amp_C
            LossScaler.has_fused_kernel = multi_tensor_applier.available
            LossScaler.multi_tensor_scale_cuda = amp_C.multi_tensor_scale
            LossScaler.multi_tensor_axpby_cuda = amp_C.multi_tensor_axpby
        else:
            if not LossScaler.warned_no_fused_kernel:
                maybe_print(
                    "Warning:  multi_tensor_applier fused unscale kernel is unavailable, "
                    "possibly because apex was installed without --cuda_ext --cpp_ext. "
                    "Using Python fallback.  Original ImportError was: " +
                    repr(multi_tensor_applier.import_err),
                    True)
            LossScaler.has_fused_kernel = False
            LossScaler.warned_no_fused_kernel = True

    def loss_scale(self):
        return self._loss_scale

    def unscale_python(self, model_grads, master_grads, scale):
        for model, master in zip(model_grads, master_grads):
            if model is not None:
                if not LossScaler.warned_unscaling_non_fp32_grad:
                    if master.dtype != torch.float32:
                        maybe_print(
                            "Attempting to unscale a grad with type {} ".format(master.type()) +
                            "Unscaling non-fp32 grads may indicate an error. "
                            "When using Amp, you don't need to call .half() on your model.")
                        LossScaler.warned_unscaling_non_fp32_grad = True
                self._has_overflow = scale_check_overflow_python(model,
                                                                 master,
                                                                 1./scale,
                                                                 self.dynamic)
                if self._has_overflow and self.dynamic:
                    break

    # unused_scale keeps some of the old API alive for hopefully a short time.
    def unscale(self, model_grads, master_grads, unused_scale, models_are_masters=False):
        if self._has_overflow:
            return

        scale = self._loss_scale

        if scale == 1.0 and models_are_masters and not self.dynamic:
            return

        if LossScaler.has_fused_kernel:
            # if (not LossScaler.warned_unscaling_non_fp32_grad
            #     and master_grads[0].dtype == torch.float16):
            #     print("Warning:  unscaling grads that are not FP32. "
            #           "Unscaling non-fp32 grads may indicate an error. "
            #           "When using Amp, you don't need to call .half() on your model.")
            #     # Setting this to True unconditionally allows the possibility of an escape
            #     # if never-before-seen non-fp32 grads are created in some later iteration.
            #     LossScaler.warned_unscaling_non_fp32_grad = True
            multi_tensor_applier(LossScaler.multi_tensor_scale_cuda,
                                 self._overflow_buf,
                                 [model_grads, master_grads],
                                 1./scale)
        else:
            self.unscale_python(model_grads, master_grads, scale)

        # Defer to update_scale
        # If the fused kernel is available, we only need one D2H memcopy and sync.
        # if LossScaler.has_fused_kernel and self.dynamic and not self._has_overflow:
        #     self._has_overflow = self._overflow_buf.item()

    def unscale_with_stashed_python(self,
                                    model_grads,
                                    stashed_master_grads,
                                    master_grads,
                                    scale):
        for model, stashed, master in zip(model_grads, stashed_master_grads, master_grads):
            if model is None and stashed is None:
                continue
            else:
                if not LossScaler.warned_unscaling_non_fp32_grad:
                    if master.dtype != torch.float32:
                        maybe_print(
                            "Attempting to unscale a grad with type {} ".format(master.type()) +
                            "Unscaling non-fp32 grads may indicate an error. "
                            "When using Amp, you don't need to call .half() on your model.")
                        LossScaler.warned_unscaling_non_fp32_grad = True
                self._has_overflow = axpby_check_overflow_python(model,
                                                                 stashed,
                                                                 master,
                                                                 1./scale,
                                                                 self.dynamic)
                if self._has_overflow and self.dynamic:
                    break

    def unscale_with_stashed(self,
                             model_grads,
                             stashed_master_grads,
                             master_grads):
        if self._has_overflow:
            return

        scale = self._loss_scale

        if LossScaler.has_fused_kernel:
            if (not LossScaler.warned_unscaling_non_fp32_grad
                and master_grads[0].dtype == torch.float16):
                print("Warning:  unscaling grads that are not FP32. "
                      "Unscaling non-fp32 grads may indicate an error. "
                      "When using Amp, you don't need to call .half() on your model.")
                # Setting this to True unconditionally allows the possibility of an escape
                # if never-before-seen non-fp32 grads are created in some later iteration.
                LossScaler.warned_unscaling_non_fp32_grad = True
            multi_tensor_applier(LossScaler.multi_tensor_axpby_cuda,
                                 self._overflow_buf,
                                 [model_grads, stashed_master_grads, master_grads],
                                 1./scale,
                                 1.0,
                                 0) # check only arg 0, aka the incoming model grads, for infs
        else:
            self.unscale_with_stashed_python(model_grads,
                                             stashed_master_grads,
                                             master_grads,
                                             scale)

        # Defer to update_scale
        # If the fused kernel is available, we only need one D2H memcopy and sync.
        # if LossScaler.has_fused_kernel and self.dynamic and not self._has_overflow:
        #     self._has_overflow = self._overflow_buf.item()

    def clear_overflow_state(self):
        self._has_overflow = False
        if self.has_fused_kernel:
            self._overflow_buf.zero_()

    # Separate so unscale() can be called more that once before updating.
    def update_scale(self):
        # If the fused kernel is available, we only need one D2H memcopy and sync.
        if LossScaler.has_fused_kernel and self.dynamic and not self._has_overflow:
            self._has_overflow = self._overflow_buf.item()

        if self._has_overflow and self.dynamic:
            should_skip = True
            if(self._min_loss_scale):
                self._loss_scale = max(self._min_loss_scale, self._loss_scale/2.)
            else:
                self._loss_scale = self._loss_scale/2.
            self._unskipped = 0
        else:
            should_skip = False
            self._unskipped += 1

        if self._unskipped == self._scale_seq_len and self.dynamic:
            self._loss_scale = min(self._max_loss_scale, self._loss_scale*2.)
            self._unskipped = 0

        return should_skip
