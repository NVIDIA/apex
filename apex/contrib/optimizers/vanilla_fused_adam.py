import types
import torch
import importlib
from apex.multi_tensor_apply import multi_tensor_applier

class VanillaFusedAdam(torch.optim.Optimizer):

    """Implements Adam algorithm. Currently GPU-only.  Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.

    This code implements a "plain vanilla" adam optimizer. There is no overlap of
    gradient reductions with bprop and no distribution of weight updates. It does
    use the undo feature. It is expected that the adam optimizer with distributed
    weight updates and the vanilla optimizer will produce binary identical results 
    given sufficient precision in the radix decomposition of the global scaler.
    One set of parameters that yields sufficient precision is -40, 40 and 2 for
    radix_min_digit, radix_max_digit and radix_base respectively.

    This code is meant for debugging only.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
        use_mt (boolean, optional): use multi tensor apply for lower launch
            latency. (default: False)
        overlap_reductions(boolean, optional): whether to overlap reductions
            with bprop (default: True)
        num_prestats (integer, optional): number of fp64 stats that will be
            reduced during first fp16 gradient reduction block. 

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params,
                 lr=1e-3, bias_correction = True,
                 betas=(0.9, 0.999), eps=1e-8, eps_inside_sqrt = False,
                 weight_decay=0., max_grad_norm=0., amsgrad=False, use_mt=False,
                 amp_scale_adjustment=1.0, overlap_reductions=True,
                 num_prestats=0, radix_min_digit=-20, radix_max_digit=20,
                 radix_base=2, num_blocks=4, full_pipeline=True,
                 normalize_by_L2_grad_norm=False, distributed_weight_update=0,
                 dwu_num_blocks=4, dwu_num_rs_pg=1, dwu_num_ar_pg=4,
                 dwu_num_ag_pg=0, dwu_num_blk_st=1):
        global fused_adam_cuda
        fused_adam_cuda = importlib.import_module("fused_adam_cuda")

        self._amp_scale_adjustment = amp_scale_adjustment

        if use_mt:
            raise RuntimeError('DistributedFusedAdam does not support use_mt.')
        if amsgrad:
            raise RuntimeError('DistributedFusedAdam does not support the AMSGrad variant.')

        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(VanillaFusedAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if  eps_inside_sqrt else 1

        self._overflow_buf = torch.cuda.IntTensor([0])

        self._global_scale = None
        
    def set_global_scale(self, global_scale):
        """Manually set global scale.
        Calling set_stats after this is an error and will raise a runtime error.

        """
        self._global_scale = global_scale

    @property
    def global_scale(self):
        return self._global_scale

    @property
    def has_overflow(self):
        """Check if overflows were detected by any call to step(...) method.
        Clears the overflow flag.
        """
        has_overflow = self._overflow_buf.item()
        self._overflow_buf.zero_()
        return has_overflow

    @property
    def peek_overflow(self):
        """Check if overflows were detected by any call to step(...) method.
        Does not clear overflow flag.
        """
        return self._overflow_buf.item()

    def _revert_step(self):
        for group in self.param_groups:
            # compute combined scale factor for this group
            combined_scale = self._global_scale
            if group['max_grad_norm'] > 0:
                # norm is in fact norm*scale
                clip = ((grad_norm / scale) + 1e-6) / group['max_grad_norm']
                if clip > 1:
                    combined_scale = clip * scale

            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']

            for p in group['params']:
                if not p.requires_grad:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                assert (len(state) > 0), "undo: state is empty"
                step = state['step']
                state['step'] = step - 1

                np = state['p'].norm()
                nm = state['m'].norm()
                nv = state['v'].norm()
                fused_adam_cuda.adam_undo(
                                     state['p'],
                                     state['p'],
                                     state['m'],
                                     state['m'],
                                     state['v'],
                                     state['v'],
                                     p.grad,
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     combined_scale,
                                     step,
                                     self.eps_mode,
                                     bias_correction,
                                     group['weight_decay'])

    def _step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self._overflow_buf.zero_()
        new_params = []
        for group in self.param_groups:
            # compute combined scale factor for this group
            combined_scale = self._global_scale
            if group['max_grad_norm'] > 0:
                # norm is in fact norm*scale
                clip = ((grad_norm / scale) + 1e-6) / group['max_grad_norm']
                if clip > 1:
                    combined_scale = clip * scale

            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']

            for p in group['params']:
                if not p.requires_grad:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                p.grad.div_(torch.distributed.get_world_size())
                torch.distributed.all_reduce(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['p'] = torch.empty_like(p).float()
                    state['m'] = torch.zeros_like(p).float()
                    state['v'] = torch.zeros_like(p).float()
                    state['p'].copy_(p.float())
                step = state['step'] + 1
                state['step'] = step

                new_p = torch.empty_like(p)
                fused_adam_cuda.adam(self._overflow_buf,
                                     state['p'],
                                     state['p'],
                                     new_p,
                                     state['m'],
                                     state['m'],
                                     state['v'],
                                     state['v'],
                                     p.grad,
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     combined_scale,
                                     step,
                                     self.eps_mode,
                                     bias_correction,
                                     group['weight_decay'])
                new_params.append((p, new_p))

        if self.peek_overflow:
            print("Reverting step")
            self._revert_step()
        else:
            with torch.no_grad():
                for p, new_p in new_params:
                    p.set_(new_p)

        return loss

    def step(self, closure=None):
        return self._step(closure)

