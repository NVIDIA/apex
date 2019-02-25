import torch
import logging
from ..multi_tensor_apply import multi_tensor_applier
from ._amp_state import _amp_state

# from apex_C import scale_check_overflow

def scale_check_overflow_python(model_grad, scale, master_grad, check_overflow=False):
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

class LossScaler(object):
    warned_no_fused_kernel = False
    warned_unscaling_non_fp32_grad = False
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
        self._overflow_buf = torch.cuda.IntTensor([0])
        if multi_tensor_applier.available:
            import amp_C
            LossScaler.has_fused_kernel = multi_tensor_applier.available
            LossScaler.multi_tensor_scale_cuda = amp_C.multi_tensor_scale
        else:
            if not LossScaler.warned_no_fused_kernel:
                print("Warning:  multi_tensor_applier fused unscale kernel is unavailable, "
                      "possibly because apex was installed without --cuda_ext --cpp_ext. "
                      "Using Python fallback.  Original ImportError was: ",
                      multi_tensor_applier.import_err)
            LossScaler.has_fused_kernel = False
            LossScaler.warned_no_fused_kernel = True

    def loss_scale(self):
        return self._loss_scale

    def unscale_grads_python(self, model_grads, master_grads, scale):
        for model, master in zip(model_grads, master_grads):
            if model is not None:
                if not LossScaler.warned_unscaling_non_fp32_grad:
                    if master.type() != "torch.cuda.FloatTensor":
                        logger = logging.getLogger("apex.amp")
                        logger.warning(
                            "Attempting to unscale a grad with type {} ".format(master.type()) +
                            "Unscaling non-fp32 grads may indicate an error. "
                            "When using Amp, you don't need to call .half() on your model.")
                        LossScaler.warned_unscaling_non_fp32_grad = True
                self._has_overflow = scale_check_overflow_python(
                    model,
                    1./scale,
                    master,
                    self.dynamic)
                if self._has_overflow and self.dynamic:
                    break

    def clear_overflow_state(self):
        self._has_overflow = False
        if self.has_fused_kernel:
            self._overflow_buf.zero_()

    def unscale(self, model_params, master_params, scale):
        # torch.cuda.nvtx.range_push("unscale")
        if self._has_overflow:
            # torch.cuda.nvtx.range_pop()
            return

        # Lots of defensive list processing going on here.  Way more less efficient than
        # consuming the iterator directly.  Need to examine Python overhead.
        model_master_params = [(model, master) for model, master
            in zip(model_params, master_params)] # some of these may be None

        # Sync the None-ness of model and master params.
        all_same = True
        for model, master in model_master_params:
            if model.grad is None and master.grad is not None:
                master.grad = None
            if model.grad is not None and master.grad is None:
                master.grad = torch.empty_like(master)
            if model.grad is not master.grad:
                all_same = False

        model_grads = [mmp[0].grad.data for mmp in model_master_params if mmp[0].grad is not None]
        master_grads = [mmp[1].grad.data for mmp in model_master_params if mmp[1].grad is not None]

        if LossScaler.has_fused_kernel:
            # The master grads should never be fp16.  The kernel can't handle that, so bail out
            # and print a warning.  This is overly conservative, and maybe we do want to enable
            # fast downscaling of fp16 grads eventually.
            if not LossScaler.warned_unscaling_non_fp32_grad:
                if any(grad.type() != "torch.cuda.FloatTensor" for grad in master_grads):
                    logger = logging.getLogger("apex.amp")
                    logger.warning(
                        "Attempting to unscale grads that are not FP32. "
                        "Unscaling non-fp32 grads may indicate an error. "
                        "When using Amp, you don't need to call .half() on your model.")
                # Warning:  setting this to True unconditionally allows the possibility of an escape
                # if never-before-seen non-fp32 grads are created in some later iteration.
                LossScaler.warned_unscaling_non_fp32_grad = True
            # handle case of opt_level O1 and loss_scale 1.0.  There's also some
            # special-cased yields in scale_loss to potentially short-circuit earlier.
            # TODO:  Profile and find out if all the O(N) list processing in unscale()
            # is a bottleneck.
            if scale == 1.0 and all_same and not self.dynamic:
                # torch.cuda.nvtx.range_pop()
                return
            else:
                multi_tensor_applier(
                    LossScaler.multi_tensor_scale_cuda,
                    self._overflow_buf,
                    [model_grads, master_grads],
                    1./scale)
        else:
            self.unscale_grads_python(model_grads, master_grads, scale)

        # If the fused kernel is available, we only need one D2H memcopy and sync.
        if LossScaler.has_fused_kernel and self.dynamic and not self._has_overflow:
            self._has_overflow = self._overflow_buf.item()

        # torch.cuda.nvtx.range_pop()

    # Separate so unscale() can be called more that once before updating.
    def update_scale(self):
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
