import math
import torch
import importlib
import amp_C
from apex.multi_tensor_apply import multi_tensor_applier

class DistributedFusedAdamV3(torch.optim.Optimizer):

    """Implements Adam algorithm. Currently GPU-only.  Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.

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
                 amp_scale_adjustment=1.0, overlap_reductions=True, full_pipeline=True,
                 compute_L2_grad_norm=False, distributed_weight_update=0,
                 dwu_group_size=0, dwu_num_blocks=4, dwu_num_rs_pg=1, dwu_num_ar_pg=4,
                 dwu_num_ag_pg=0, revert_method=1, flat_mt=False,
                 dwu_num_chunks=4, predivide=True, e5m2_allgather=False,
                 do_not_flatten_model=False):
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
        super(DistributedFusedAdamV3, self).__init__(params, defaults)
        self.eps_mode = 0 if  eps_inside_sqrt else 1

        self._overflow_buf = torch.cuda.IntTensor([0])

        assert (len(self.param_groups) == 1), "More than one parameter group is not supported."

        # Way to revert a step
        # 3 -> undo kernel + double buffer (debug, print norm of difference)
        # 2 -> double buffer fp32 parameters
        # 1 -> undo kernel
        self._revert_method = revert_method
        if self._revert_method > 1:
            print("revert_method -> double buffer fp32 parameters, will consume more memory")

        self._last_step = False
        self._overlap_reductions = overlap_reductions
        self._global_scale = None
        self._num_blocks = dwu_num_blocks
        self._predivide = predivide
        self._e5m2_allgather = e5m2_allgather
        self._do_not_flatten_model = do_not_flatten_model
        self._full_pipeline = full_pipeline
        self._L2_grad_norm = None
        self._group_size = torch.cuda.device_count() if dwu_group_size <= 0 else dwu_group_size
        self._world_size = torch.distributed.get_world_size()
        self._num_groups = self._world_size // self._group_size
        self._rank_in_group = torch.distributed.get_rank() % self._group_size

        p_offset = 0
        p_i = 0
        self._param_state = None
        self._model_params = []
        self._grads_info = []
        self._grad_accs = []
        for group in self.param_groups:
            self._param_group = group
            prev = None
            for p in group['params']:
                torch.distributed.broadcast(p,0)
                if not p.requires_grad:
                    continue
                self._model_params.append(p)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                if self._param_state is None:
                    self._param_state = state
                p_grads_size = p.numel()
                def wrapper(param, param_i, param_grads_size, param_offset):
                    param_tmp = param.expand_as(param)
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    def allreduce_hook(*unused):
                        self._do_overlapped_reduction(param_i, param_grads_size, param_offset, param)
                    grad_acc.register_hook(allreduce_hook)
                    self._grad_accs.append(grad_acc)
                self._grads_info.append({"param_grads_size":p_grads_size, "param_offset":p_offset})
                wrapper(p, p_i, p_grads_size, p_offset)
                p_offset += p_grads_size
                # Only enforce 128b alignment (64 * fp16) for non-consecutive parameters
                # RNN is one example of consecutive parameters:
                # (weight_ih, weight_hh, bias_ih, bias_hh)
                if prev is not None and (prev.data_ptr() + prev.numel() * prev.element_size() != p.data_ptr()):
                    p_offset = ((p_offset + 63) // 64) * 64
                prev = p
                p_i += 1
        self._grads_generated = [False]*len(self._grads_info)
        self._flat_mt = flat_mt
        self._grads = []
        self._current_block = self._num_blocks

        self._net_total_param_size = p_offset
        self._total_param_size = p_offset
        dwu_min_page_size = 256 * self._num_blocks * self._group_size
        self._total_param_size = ((self._total_param_size + dwu_min_page_size - 1) // dwu_min_page_size) * dwu_min_page_size
        self._block_size = self._total_param_size // self._num_blocks
        self._shard_size = self._total_param_size // self._group_size
        print("self._net_total_param_size=%d, self._total_param_size=%d, dwu_min_page_size=%d, self._block_size=%d, self._shard_size=%d" % (self._net_total_param_size, self._total_param_size,dwu_min_page_size,self._block_size,self._shard_size))

        self._low_param_i = [0]*self._num_blocks
        for block_id in range(self._num_blocks-1,-1,-1):
            p_i = len(self._grads_info)-1
            while p_i > 0 and self._grads_info[p_i]["param_offset"] > block_id*self._block_size:
                p_i -= 1
            self._low_param_i[block_id] = p_i
        print(self._low_param_i)

        self._flat_grads = torch.zeros([self._total_param_size], dtype=torch.float16, device='cuda')
        self._flat_params = torch.zeros_like(self._flat_grads)

        def _flat_split(flat):
            def __flat_blockify(flat):
                return [flat[block_id*self._block_size:(block_id+1)*self._block_size] for block_id in range(self._num_blocks)]
            def __flat_shardify(flat):
                return [flat[shard_id*self._shard_size:(shard_id+1)*self._shard_size] for shard_id in range(self._group_size)]
            return __flat_blockify(flat), __flat_shardify(flat)
        self._flat_grads_blocks, self._flat_grads_shards = _flat_split(self._flat_grads)
        self._flat_params_blocks, self._flat_params_shards = _flat_split(self._flat_params)

        # master params
        self._fp32_p = torch.zeros([self._shard_size], dtype=torch.float32, device='cuda')
        self._fp32_m = torch.zeros([self._shard_size], dtype=torch.float32, device='cuda')
        self._fp32_v = torch.zeros([self._shard_size], dtype=torch.float32, device='cuda')

        # copy model params to flat_params and set_ model params to flat_params.
        self._individual_flat_grads = []
        with torch.no_grad():
            for p, grads_info in zip(self._model_params, self._grads_info):
                start = grads_info["param_offset"]
                end = start + grads_info["param_grads_size"]
                flat_p = self._flat_params[start:end].view_as(p)
                flat_p.copy_(p)
                p.set_(flat_p)
                flat_grad = self._flat_grads[start:end]
                self._individual_flat_grads.append(flat_grad)
        self._fp32_p.copy_(self._flat_params_shards[self._rank_in_group].float())

        self._dwu_st = torch.cuda.Stream()
        self._l2_grad_norm_st = torch.cuda.Stream()
        for group_i in range(self._num_groups):
            ranks = [group_i*self._group_size+local_rank for local_rank in range(self._group_size)]
            pg = torch.distributed.new_group(ranks=ranks)
            if torch.distributed.get_rank() in ranks:
                self._ag_pg = pg
                torch.distributed.all_reduce(self._overflow_buf, group=self._ag_pg)

        import inspect
        assert ('no_copy' in inspect.getfullargspec(torch.distributed.reduce_scatter).args), "This version of c10d does not support no_copy option"

    @property
    def has_overflow(self):
        return True if not self.L2_grad_norm is None and not math.isfinite(self.L2_grad_norm) else False

    def set_last_step(self, last_step):
        self._last_step = last_step
        
    def _get_flush_block(self):
        flush_block = []
        if self._current_block > 0 and self._grads_generated[self._low_param_i[self._current_block-1]]:
            num_grads = len(self._grads_generated)
            contiguous_idx = num_grads
            while contiguous_idx > 0 and self._grads_generated[contiguous_idx-1]:
                contiguous_idx -= 1

            if contiguous_idx < num_grads and self._grads_info[contiguous_idx]["param_offset"] <= (self._current_block-1)*self._block_size:
                self._current_block -= 1
                start = self._current_block * self._block_size
                end = (self._current_block+1) * self._block_size
                flush_block = [start, end]

        return flush_block

    def __launch_step_kernel(self, p, p_copy, m, v, g):
        combined_scale = self._global_scale
        if self._param_group['max_grad_norm'] > 0 and math.isfinite(self.L2_grad_norm):
            combined_scale = self._param_group['max_grad_norm'] / (self.L2_grad_norm / self._global_scale + 1e-6)
            combined_scale = self._global_scale / min(1, combined_scale)
        bias_correction = 1 if self._param_group['bias_correction'] else 0
        beta1, beta2 = self._param_group['betas']
        fused_adam_cuda.reversible_adam(
                p, p_copy, m, v, g,
                self._param_group['lr'],
                beta1,
                beta2,
                self._param_group['eps'],
                combined_scale,
                self._param_state['step']+1,
                self.eps_mode,
                bias_correction,
                self._param_group['weight_decay'])

    def _flatten_grad_mt(self, scale):
        if self._flat_mt and len(self._grads) > 0:
            self._overflow_buf.zero_()
            multi_tensor_applier(
                    amp_C.multi_tensor_scale,
                    self._overflow_buf,
                    list(zip(*self._grads)),
                    scale)
            self._grads = []

    def _do_overlapped_reduction(self, param_i, param_grads_size, param_offset, param):
        # handle overlapped reductions
        if self._flat_mt:
            self._grads.append( (param.grad, self._individual_flat_grads[param_i]) )
        else:
            torch.div(param.grad, self._world_size if self._predivide else 1.0, out=self._individual_flat_grads[param_i])
        self._grads_generated[param_i]=True
        if not self._last_step and self._overlap_reductions:
            flush_block = self._get_flush_block()
            while flush_block:
                block_id = flush_block[0] // self._block_size
                self._dwu_st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._dwu_st):
                    self._flatten_grad_mt(1.0/self._world_size if self._predivide else 1.0)
                    torch.distributed.all_reduce(self._flat_grads_blocks[block_id])
                if block_id == 0:
                    self._l2_grad_norm_st.wait_stream(self._dwu_st)
                    with torch.cuda.stream(self._l2_grad_norm_st):
                        self._L2_grad_norm = self._flat_grads.norm(dtype=torch.float32, p=2).item()
                flush_block = self._get_flush_block()

    def set_global_scale(self, global_scale):
        """Set global scale.
        """
        self._global_scale = global_scale

    @property
    def global_scale(self):
        return self._global_scale

    @property
    def L2_grad_norm(self):
        torch.cuda.current_stream().wait_stream(self._l2_grad_norm_st)
        return self._L2_grad_norm

    def complete_reductions(self):
        """Complete reductions if full pipeline is not selected or overlap is not allowed.
        """

        if self._last_step:
            # zero out gradients that have not been completed yet
            for param_i, flat_grad in enumerate(self._individual_flat_grads):
                if not self._grads_generated[param_i]:
                    flat_grad.zero_()
                    self._grads_generated[param_i] = True

        if self._last_step or not self._overlap_reductions:
            # nothing done so far, run full pipeline after reductions
            self._dwu_st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._dwu_st):
                self._flatten_grad_mt(1.0/self._world_size if self._predivide else 1.0)
                torch.distributed.all_reduce(self._flat_grads)
            self._l2_grad_norm_st.wait_stream(self._dwu_st)
            with torch.cuda.stream(self._l2_grad_norm_st):
                self._L2_grad_norm = self._flat_grads.norm(dtype=torch.float32, p=2).item()

        self._current_block = self._num_blocks
        self._grads_generated = [False]*len(self._grads_info)

    def step(self, closure=None, skip_overflow_check=False):
        loss = None
        if closure is not None:
            loss = closure()

        with torch.cuda.stream(self._dwu_st):
            self.__launch_step_kernel(
                self._fp32_p,
                self._flat_params_shards[self._rank_in_group],
                self._fp32_m,
                self._fp32_v,
                self._flat_grads_shards[self._rank_in_group])
            torch.distributed.all_gather(self._flat_params_shards, self._flat_params_shards[self._rank_in_group], group=self._ag_pg, no_copy=True)
            for p in self._model_params: self.state[p]['step'] += 1

        torch.cuda.current_stream().wait_stream(self._dwu_st)

        return loss


