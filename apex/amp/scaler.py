import torch
import logging
from ..multi_tensor_apply import multi_tensor_applier
from ._amp_state import _amp_state
from itertools import product

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
        if self._has_overflow:
            return

        # Lots of defensive list processing going on here.  Way more less efficient than
        # consuming the iterator directly.  Need to examine Python overhead.
        model_master_params = [(model, master) for model, master
            in zip(model_params, master_params)] # some of these may be None

        if LossScaler.has_fused_kernel:
            # TODO:  Make these lists permanent attributes of self, so they don't need to be created
            # or garbage collected.  Profiler shows that garbage collection overhead may be
            # substantial (200-300 usec).
            # This may be tricky because right now the lists need to be packed densely.
            # Maybe this could be handled within the multi_tensor_apply wrapper
            # (allow some Tensors to be None using at::optional).
            src_dst_pairs = {torch.float16 : {torch.float16 : [[],[]], torch.float32 : [[],[]]},
                             torch.float32 : {torch.float16 : [[],[]], torch.float32 : [[],[]]}}

            for model, master in model_master_params:
                # Sync the None-ness of model and master params
                if model.grad is None and master.grad is not None:
                    master.grad = None
                if model.grad is not None and master.grad is None:
                    master.grad = torch.empty_like(master)

                if model.grad is not None:
                    if model.grad is master.grad and scale == 1.0 and not self.dynamic:
                        continue
                    else:
                        src_dst_pairs[model.dtype][master.dtype][0].append(model.grad.data)
                        src_dst_pairs[model.dtype][master.dtype][1].append(master.grad.data)

            assert len(src_dst_pairs[torch.float32][torch.float16][0]) == 0, "The loss scaler is "\
                "being asked to unscale FP32 model gradients into FP16 master gradients.  This is "\
                "almost certainly an error."

            for src, dst in product((torch.float16, torch.float32),
                                    (torch.float16, torch.float32)):
                if len(src_dst_pairs[src][dst][0]) > 0:
                    if not LossScaler.warned_unscaling_non_fp32_grad and dst is torch.float16:
                        print("Warning:  unscaling grads that are not FP32. "
                              "Unscaling non-fp32 grads may indicate an error. "
                              "When using Amp, you don't need to call .half() on your model.")
                        # Setting this to True unconditionally allows the possibility of an escape
                        # if never-before-seen non-fp32 grads are created in some later iteration.
                        LossScaler.warned_unscaling_non_fp32_grad = True
                    multi_tensor_applier(
                        LossScaler.multi_tensor_scale_cuda,
                        self._overflow_buf,
                        src_dst_pairs[src][dst],
                        1./scale)
        else:
            # Sync the None-ness of model and master params.
            all_same = True
            for model, master in model_master_params:
                if model.grad is None and master.grad is not None:
                    master.grad = None
                if model.grad is not None and master.grad is None:
                    master.grad = torch.empty_like(master)
                if model.grad is not master.grad:
                    all_same = False

            if scale == 1.0 and all_same and not self.dynamic:
                return

            # TODO:  Make these lists permanent attributes of self, so they don't need to be created
            # or garbage collected?
            model_grads = [mmp[0].grad.data for mmp in model_master_params if mmp[0].grad is not None]
            master_grads = [mmp[1].grad.data for mmp in model_master_params if mmp[1].grad is not None]

            self.unscale_grads_python(model_grads, master_grads, scale)

        # If the fused kernel is available, we only need one D2H memcopy and sync.
        if LossScaler.has_fused_kernel and self.dynamic and not self._has_overflow:
            self._has_overflow = self._overflow_buf.item()

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
