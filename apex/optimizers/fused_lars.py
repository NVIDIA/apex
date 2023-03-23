import torch
from torch.optim.optimizer import Optimizer, required
from torch import nn
from torch.nn.parameter import Parameter
from apex.multi_tensor_apply import multi_tensor_applier

class FusedLARS(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, trust_coefficient=0.001, eps=0.0,
                 nesterov=False, wd_after_momentum=False,
                 materialize_master_grads=True, set_grad_none=False):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, trust_coefficient=trust_coefficient, eps=eps, is_skipped=False)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FusedLARS, self).__init__(params, defaults)

        self.wd_after_momentum = wd_after_momentum
        self.materialize_master_grads = materialize_master_grads
        self.most_recent_scale = 1.0
        self.scale_set_by_backward = False
        self.set_grad_none = set_grad_none
        self.trust_coefficient = trust_coefficient
        self.eps = eps

        if multi_tensor_applier.available:
            import amp_C
            # Skip buffer
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=self.param_groups[0]["params"][0].device)
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
            self.multi_tensor_lars = amp_C.multi_tensor_lars
            self._dummy_overflow_buf = torch.cuda.IntTensor(1).zero_()
        else:
            raise RuntimeError('apex.optimizers.FusedLARS requires cuda extensions')
        
    def __setstate__(self, state):
        super(FusedLARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedLARS, self).zero_grad()

    def get_momentums(self, params):
        momentums = []
        first_run = True
        for p in params:
            if p.grad is None:
                continue

            param_state = self.state[p]
            d_p = p.grad.data
            # torch.optim.SGD initializes momentum in the main loop, we have
            # to do it here, and track whether or not we've done so, so that
            # momentum application can be skipped in the main kernel.
            if 'momentum_buffer' not in param_state:
                first_run = True
                buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                momentums.append(buf)
            else:
                first_run = False
                momentums.append(param_state['momentum_buffer'])
        return momentums, first_run

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        explicit_master_params = (hasattr(self, "_amp_stash") and
                                  hasattr(self._amp_stash, "fp32_from_fp16_groups"))
        explicit_master_params = False

        for gid, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            is_skipped = group['is_skipped']
            
            # For each group, there are 3 possible combinations we need to consider:
            # grad_type, param_to_update_type, momentum_type, requires_fp16_model_copy
            # 1. fp16, fp16, fp16, No
            # 2. fp32, fp32, fp32, No
            # 3. fp16, fp32, fp32, Yes

            first_runs = [True, True]
            g_norms_grp = []
            w_norms_grp = []


            # I think a bit of code divergence in exchange for naming clarity is worthwhile
            if explicit_master_params:
                print('explicit_master_params')
                stash = self._amp_stash

                fp32_params = [p for p in stash.fp32_from_fp32_groups[gid] if p.grad is not None]
                fp32_grads = [p.grad for p in stash.fp32_from_fp32_groups[gid] if p.grad is not None]
                fp32_momentums, first_runs[1] = self.get_momentums(fp32_params)

                if self.materialize_master_grads:
                    fp16_model_params = [p for i, p in enumerate(
                        stash.fp16_groups[gid]) if stash.fp32_from_fp16_groups[gid][i].grad is not None]
                    fp32_from_fp16_grads = [p.grad for p in stash.fp32_from_fp16_groups[gid] if p.grad is not None]
                    fp32_from_fp16_params = [p for p in stash.fp32_from_fp16_groups[gid] if p.grad is not None]
                    fp32_from_fp16_momentums, first_runs[0] = self.get_momentums(fp32_from_fp16_params)

                    fp16_set = [fp32_from_fp16_grads, fp32_from_fp16_params,
                                fp32_from_fp16_momentums, fp16_model_params]
                else:
                    fp16_model_params = [p for p in stash.fp16_groups[gid] if p.grad is not None]
                    fp16_model_grads = [p.grad for p in stash.fp16_groups[gid] if p.grad is not None]
                    fp32_from_fp16_params = [p for i, p in enumerate(
                        stash.fp32_from_fp16_groups[gid]) if stash.fp16_groups[gid][i].grad is not None]
                    fp32_from_fp16_momentums, first_runs[0] = self.get_momentums(fp32_from_fp16_params)

                    fp16_set = [fp16_model_grads, fp32_from_fp16_params,
                                fp32_from_fp16_momentums, fp16_model_params]

                launch_sets= [fp16_set, [fp32_grads, fp32_params, fp32_momentums]]

            else:
                fp16_params = [p for p in group['params'] if (p.dtype == torch.float16 and p.grad is not None)]
                #fp16_grads = [p.grad for p in group['params'] if (p.dtype == torch.float16 and p.grad is not None)]
                fp16_grads = []
                for p in fp16_params:
                    if p.is_contiguous():
                        fp16_grads.append(p.grad)
                    elif p.is_contiguous(memory_format=torch.channels_last):
                        fp16_grads.append(p.grad.to(memory_format=torch.channels_last))
                fp16_momentums, first_runs[0] = self.get_momentums(fp16_params)
                # Compute L2 norms
                if len(fp16_params) > 0:
                    w_norms = multi_tensor_applier(
                            self.multi_tensor_l2norm,
                            self._dummy_overflow_buf,
                            [[p.data for p in fp16_params]],
                            True)[1]
                    g_norms = multi_tensor_applier(
                            self.multi_tensor_l2norm,
                            self._dummy_overflow_buf,
                            [[p.data for p in fp16_grads]],
                            True)[1]
                else:
                    w_norms = []
                    g_norms = []
                w_norms_grp.append(w_norms)
                g_norms_grp.append(g_norms)

                fp32_params = [p for p in group['params'] if (p.dtype == torch.float32 and p.grad is not None)]
                fp32_grads = []
                for p in fp32_params:
                    if p.is_contiguous():
                        fp32_grads.append(p.grad)
                    elif p.is_contiguous(memory_format=torch.channels_last):
                        fp32_grads.append(p.grad.to(memory_format=torch.channels_last))
                fp32_momentums, first_runs[1] = self.get_momentums(fp32_params)
                # Compute L2 norms
                if len(fp32_params) > 0:
                    w_norms = multi_tensor_applier(
                            self.multi_tensor_l2norm,
                            self._dummy_overflow_buf,
                            [[p.data for p in fp32_params]],
                            True)[1]
                    g_norms = multi_tensor_applier(
                            self.multi_tensor_l2norm,
                            self._dummy_overflow_buf,
                            [[p.data for p in fp32_grads]],
                            True)[1]
                else:
                    w_norms = []
                    g_norms = []
                w_norms_grp.append(w_norms)
                g_norms_grp.append(g_norms)

                launch_sets = [[fp16_grads, fp16_params, fp16_momentums],
                               [fp32_grads, fp32_params, fp32_momentums]]

            for s, (launch_set, first_run, g_norms, w_norms) in enumerate(zip(launch_sets, first_runs, g_norms_grp, w_norms_grp)):
                assert len(launch_set[0]) == len(launch_set[1])
                assert len(launch_set[0]) == len(launch_set[2])
                if len(launch_set[0]) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_lars,
                        self._dummy_overflow_buf,
                        launch_set,
                        g_norms,
                        w_norms,
                        group['lr'],
                        group['trust_coefficient'],
                        self.eps,
                        weight_decay,
                        momentum,
                        dampening,
                        nesterov,
                        first_run,
                        self.wd_after_momentum,
                        1.0/self.most_recent_scale,
                        group['is_skipped'])

        self.most_recent_scale = 1.0
        self.scale_set_by_backward = False

        return loss
