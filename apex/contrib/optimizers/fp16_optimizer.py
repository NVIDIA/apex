import torch
from apex.multi_tensor_apply import multi_tensor_applier

class FP16_Optimizer(object):
    """
    :class:`FP16_Optimizer` A cutdown version of apex.fp16_utils.FP16_Optimizer.
    Designed only to wrap apex.contrib.optimizers.FusedAdam, FusedSGD.
    Refer to apex.fp16_utils documents for more information.
    Example::
        model = torch.nn.Linear(D_in, D_out).cuda().half()
        optimizer = apex.contrib.optimizers.FusedSGD(model.parameters())
        optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
        ...
        # loss.backward() becomes:
        optimizer.backward(loss)
        ...
    Example with dynamic loss scaling::
        ...
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                                   # optional arg to control dynamic loss scaling behavior
                                   # dynamic_loss_args={'scale_window' : 500})
                                   # Usually, dynamic_loss_args is not necessary.
    """

    def __init__(self,
                 init_optimizer,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True):

        print("\nThis fp16_optimizer is designed to only work with apex.contrib.optimizers.*")
        print("To update, use updated optimizers with AMP.")
        # The fused optimizer does all the work. We need this layer for two reason:
        # 1. maintain same user API from apex.fp16_utils
        # 2. keep common stuff here in case we need to add new fused optimizer later

        if not torch.cuda.is_available:
            raise SystemError("Cannot use fp16 without CUDA.")
        self.optimizer = init_optimizer

        self.fp16_groups = [] # model params
        self.fp32_groups = [] # master weights

        # iterate over param_groups
        for param_group in self.optimizer.param_groups:
            fp16_group = []
            fp32_group = []
            for p in param_group['params']:
                fp16_group.append(p)
                fp32_group.append(p.clone().float().detach())
            self.fp16_groups.append(fp16_group)
            self.fp32_groups.append(fp32_group)
            param_group['params'] = fp32_group

        if multi_tensor_applier.available:
            import amp_C
            self.overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_l2norm=amp_C.multi_tensor_l2norm
        else:
            raise RuntimeError('FP16_Optimizer requires cuda extensions')

        # we may have a way of fusing dynamic scale. Do not support for now
        if dynamic_loss_scale:
            if dynamic_loss_args is not None:
                raise SystemError("Do not support dynamic loss scale args for now.")
            self.dynamic_loss_scale = True
            self.cur_scale = 2**16
            self.cur_iter = 0
            self.last_overflow_iter = -1
            self.scale_factor = 2
            self.scale_window = 1000
        else:
            self.dynamic_loss_scale = False
            self.cur_iter = 0
            self.cur_scale = static_loss_scale
        self.verbose = verbose

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def step(self, closure=None):
        """
        Not supporting closure.
        """
        fp16_grads = []
        norm_groups = []
        skip = False

        for group in self.fp16_groups:
            fp16_grad = []
            for i, p in enumerate(group):
                fp16_grad.append(p.grad)
            fp16_grads.append(fp16_grad)
        
        # nan check
        self.overflow_buf.zero_()
        for fp16_grad in fp16_grads:
            if len(fp16_grad) > 0:
                norm, norm_per_tensor = multi_tensor_applier(self.multi_tensor_l2norm,
                                                             self.overflow_buf,
                                                             [fp16_grad], True)
                norm_groups.append(norm)
                if self.overflow_buf.item() != 0:
                    skip = True

        if skip:
            self._update_scale(skip)
            return

        # norm is in fact norm*cur_scale
        self.optimizer.step(grads=fp16_grads,
                            output_params=self.fp16_groups,
                            scale=self.cur_scale,
                            grad_norms=norm_groups)

        self._update_scale(False)
        return

    def backward(self, loss):
        """
        :attr:`backward` performs the following steps:
        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        scaled_loss = (loss.float()) * self.cur_scale
        scaled_loss.backward()

    def _update_scale(self, skip):
        if self.dynamic_loss_scale:
            if skip:
                if self.verbose:
                    print("\nGrad overflow on iteration", self.cur_iter)
                    print("Using dynamic loss scale of", self.cur_scale)
                self.cur_scale = max(self.cur_scale/self.scale_factor, 1)
                self.last_overflow_iter = self.cur_iter
            else:
                if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                    self.cur_scale *= self.scale_factor
        else:
            if skip:
                print("\nGrad overflow on iteration", self.cur_iter)
                print("Using static loss scale of", self.cur_scale)
        self.cur_iter +=1
        return

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['cur_scale'] = self.cur_scale
        state_dict['cur_iter'] = self.cur_iter
        if state_dict['dynamic_loss_scale']:
            state_dict['last_overflow_iter'] = self.last_overflow_iter
            state_dict['scale_factor'] = self.scale_factor
            state_dict['scale_window'] = self.scale_window
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['fp32_groups'] = self.fp32_groups
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        # I think it should actually be ok to reload the optimizer before the model.
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        if state_dict['dynamic_loss_scale']:
            self.last_overflow_iter = state_dict['last_overflow_iter']
            self.scale_factor = state_dict['scale_factor']
            self.scale_window = state_dict['scale_window']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        # At this point, the optimizer's references to the model's fp32 parameters are up to date.
        # The optimizer's hyperparameters and internal buffers are also up to date.
        # However, the fp32 master copies of the model's fp16 params stored by the optimizer are still
        # out of date.  There are two options.
        # 1:  Refresh the master params from the model's fp16 params.
        # This requires less storage but incurs precision loss.
        # 2:  Save and restore the fp32 master copies separately.
        # We choose option 2.
        #
        # Pytorch Optimizer.load_state_dict casts saved buffers (e.g. momentum) to the type and device
        # of their associated parameters, because it's possible those buffers might not exist yet in
        # the current optimizer instance.  In our case, as long as the current FP16_Optimizer has been
        # constructed in the same way as the one whose state_dict we are loading, the same master params
        # are guaranteed to exist, so we can just copy_() from the saved master params.
        for current, saved in zip(self.fp32_groups, state_dict['fp32_groups']):
            for _current, _saved in zip(current, saved):
                _current.data.copy_(_saved.data)
