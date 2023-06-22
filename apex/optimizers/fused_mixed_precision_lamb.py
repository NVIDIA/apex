import torch
from copy import deepcopy
from itertools import chain
from collections import defaultdict, abc as container_abcs

from apex.multi_tensor_apply import multi_tensor_applier

class FusedMixedPrecisionLamb(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, step=0, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01,
                 amsgrad=False, adam_w_mode=True,
                 grad_averaging=True, max_grad_norm=1.0, use_nvlamb=False,
                 reduced_precision_dtype=None):

        if amsgrad:
            raise RuntimeError('FusedLAMB does not support the AMSGrad variant.')

        # init defaults
        defaults = dict(lr=torch.tensor(lr, dtype=torch.float32),
                        step=torch.tensor([step], dtype=torch.int),
                        bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        grad_averaging=grad_averaging,
                        max_grad_norm=max_grad_norm)

        # init base module
        super(FusedMixedPrecisionLamb, self).__init__(params, defaults)

        # The learning rate (lr) and optimizer step (step) should be located on device
        # in order to faciliated device sync free execution
        device = self.param_groups[0]['params'][0].device
        tensor_state = ['lr', 'step']
        for idx,group in enumerate(self.param_groups):
            for item in tensor_state:
                self.param_groups[idx][item] = group[item].to(device=device)

        if multi_tensor_applier.available:
            import amp_C
            self.multi_tensor_l2norm=amp_C.multi_tensor_l2norm_mp
            # Skip buffer
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=device)
            self.multi_tensor_lamb = amp_C.multi_tensor_lamb_mp
        else:
            raise RuntimeError('apex.optimizers.FusedLAMB requires cuda extensions')

        # Mixed Precision support
        self.reduced_precision_dtype = reduced_precision_dtype
        self.param_groups_full_precision = []
        
        self._step_supports_amp_scaling = True
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.use_nvlamb = use_nvlamb

    # This method is overridden from the parent class because there is not a way to override
    # the nested function cast() that copies a saved piece of state to the device without
    # redundantly doing the copy.
    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = {old_id: p for old_id, p in
                  zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups)))}

        def cast(param, value):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # The original version casted the saved value to the params dtype
                # This doesn't work for mixed precision Lamb where the momentum and
                # velocity are expected to be in full precision while the params are
                # in reduced precision
                value = value.to(value.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def _setup_full_precision_params(self):
        for i, pg in enumerate(self.param_groups):
            param_list = pg['params']
            self.param_groups_full_precision.append({
                'params': [
                    p.clone().detach().to(dtype=torch.float32)
                    if (self.reduced_precision_dtype is not None) and (p.dtype == self.reduced_precision_dtype)
                    else None
                    for p in param_list
                ],
            })
   
    # add_param_groups() is overridden because default items can be tensors. The
    # parent version does not clone the default item, so two param groups can 
    # accidentally point to the same default item value where they can differ
    # given they are in separate groups.
    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        for name, default in self.defaults.items():
            if isinstance(default, torch.Tensor):
                self.param_groups[len(self.param_groups) - 1][name] = default.clone()

    @torch.no_grad()
    def step(self, closure=None, grad_scaler=None):
        loss = None
        if closure is not None:
            loss = closure()

        # The full precision params are set up in the first step of the optimizer
        # instead of in the constructor because the full precision params will get out
        # out of sync with the model params if DDP syncs the model params across devices
        # after the optimizer is constructed.
        if len(self.param_groups_full_precision) == 0 :
            self._setup_full_precision_params()

        # create separate grad lists for params
        grad_list = []
        for gid,group in enumerate(self.param_groups):
            for pid,p in enumerate(group['params']):
                assert group['params'][0].dtype == p.dtype, \
                    "Error: Parameters are not of the identical type: {} != {}".format(
                    group['params'][0].dtype, p.dtype)
                if p.grad is None:
                    continue
                grad_list.append(p.grad)
       
        # Overflow check of gradients
        device = self.param_groups[0]["params"][0].device
        found_inf = (
            grad_scaler._check_inf_per_device(self)[device]
            if grad_scaler is not None else torch.zeros((1,), device=device)
        )
        self._dummy_overflow_buf.copy_(found_inf)

        # Get unscale scale factor
        scale, inv_scale = None, None
        if grad_scaler:
            scale = grad_scaler._get_scale_async()
            inv_scale = scale.double().reciprocal().float()
        else:
            scale = torch.ones((1,), device=device)
            inv_scale = torch.ones((1,), device=device)
        
        # grad_norm is of scaled gradients.
        # So, multiply `max_grad_norm` by scale.
        max_grad_norm = self.defaults['max_grad_norm'] * scale
        grad_norm = multi_tensor_applier(
            self.multi_tensor_l2norm,
            self._dummy_overflow_buf,
            [grad_list],
            False,
        )[0]

        # Run LAMB optimization math
        for gid, (group, group_full) in enumerate(zip(self.param_groups, self.param_groups_full_precision)):
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            group['step'] += (self._dummy_overflow_buf != 1).to(torch.int)

            state_lists = [ [], # (0) grads
                            [], # (1) params
                            [], # (2) momentum state
                            [], # (3) velocity state
                          ]
            if self.reduced_precision_dtype is not None:
                state_lists.append([]) # (4) params reduced_dtype


            for p, p_full in zip(group['params'], group_full['params']):
                if p.grad is None:
                    continue
                assert not p.grad.is_sparse

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    dtype = p.dtype
                    if self.reduced_precision_dtype is not None and p.dtype == self.reduced_precision_dtype :
                        dtype = torch.float32
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=dtype)
                    # Exponential moving average of gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=dtype)

                if self.reduced_precision_dtype is not None :
                    state_lists[0].append(p.grad.data)
                    state_lists[1].append(p_full.data)
                    state_lists[2].append(state['exp_avg'])
                    state_lists[3].append(state['exp_avg_sq'])
                    state_lists[4].append(p.data)
                else :
                    state_lists[0].append(p.grad.data)
                    state_lists[1].append(p.data)
                    state_lists[2].append(state['exp_avg'])
                    state_lists[3].append(state['exp_avg_sq'])

            multi_tensor_applier(
                self.multi_tensor_lamb,
                self._dummy_overflow_buf,
                state_lists,
                group['lr'],
                beta1,
                beta2,
                group['eps'],
                group['step'],
                bias_correction,
                group['weight_decay'],
                grad_averaging,
                self.adam_w_mode,
                grad_norm,
                max_grad_norm,
                self.use_nvlamb,
                found_inf,
                inv_scale)

        return loss
