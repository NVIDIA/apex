import types
import torch
import importlib
from apex.multi_tensor_apply import multi_tensor_applier

def fused_norm(input):
    if input.type() == 'torch.cuda.HalfTensor':
        return lib.THCudaHalfTensor_normall(torch.cuda._state_cdata, input._cdata, 16384) #yes, 16384 is half 2 if you stare at it long enough
    else:
        return input.norm()

class DistributedFusedAdam(torch.optim.Optimizer):

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
                 amp_scale_adjustment=1.0, overlap_reductions=True,
                 num_prestats=0, radix_min_digit=-20, radix_max_digit=20,
                 radix_base=2, num_blocks=4, full_pipeline=True,
                 normalize_by_L2_grad_norm=False, distributed_weight_update=0,
                 dwu_num_blocks=4, dwu_num_rs_pg=1, dwu_num_ar_pg=4,
                 dwu_num_ag_pg=0):
        global fused_adam_cuda, radix_decomp_cuda
        fused_adam_cuda = importlib.import_module("fused_adam_cuda")
        radix_decomp_cuda = importlib.import_module("radix_decomp_cuda")
        # To-Do: Add radix decomp args to fairseq adam optimizer

        self._amp_scale_adjustment = amp_scale_adjustment

        if use_mt:
            raise RuntimeError('DistributedFusedAdam does not support use_mt.')
        if amsgrad:
            raise RuntimeError('DistributedFusedAdam does not support the AMSGrad variant.')

        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(DistributedFusedAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if  eps_inside_sqrt else 1

        self._dummy_overflow_buf = torch.cuda.IntTensor([0])

        self._last_step = False
        self._overlap_reductions = overlap_reductions
        self._radix_min_digit = radix_min_digit
        self._radix_max_digit = radix_max_digit
        self._radix_size = self._radix_max_digit - self._radix_min_digit + 1
        self._radix_base = radix_base
        self._num_prestats = num_prestats
        if self._num_prestats > 0:
            self._prestats_size = self._num_prestats * self._radix_size
        self._stats = None
        self._decomp_stats = None
        self._global_scale = None
        self._num_blocks = dwu_num_blocks
        if normalize_by_L2_grad_norm and full_pipeline:
            raise RuntimeError('DistributedFusedAdam does not support both full_pipeline and normalize_by_L2_grad_norm')
        self._full_pipeline = full_pipeline
        self._normalize_by_L2_grad_norm = normalize_by_L2_grad_norm
        if self._normalize_by_L2_grad_norm:
            self._partial_L2_grad_norm_squared = torch.zeros([1]).float().cuda()
        self._L2_grad_norm = None
        self._group_size = torch.cuda.device_count()
        self._world_size = torch.distributed.get_world_size()
        self._num_groups = self._world_size // self._group_size
        self._rank_in_group = torch.distributed.get_rank() % self._group_size

        p_offset = 0
        p_i = 0
        if self._overlap_reductions:
            self._grads_info = []
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                p_grads_size = p.numel()
                if self._overlap_reductions:
                    def wrapper(param, param_i, param_grads_size, param_offset):
                        def allreduce_hook(grad):
                            self._do_overlapped_reduction(param_i, param_grads_size, param_offset, grad)
                        param.register_hook(allreduce_hook)
                    self._grads_info.append({"param_grads_size":p_grads_size, "param_offset":p_offset})
                    wrapper(p, p_i, p_grads_size, p_offset)
                p_offset += p_grads_size
                p_i += 1
        self._grads_generated = [False]*len(self._grads_info)
        self._current_block = self._num_blocks
        self._events = []

        self._total_param_size = p_offset
        dwu_min_page_size = 256 * self._num_blocks * self._group_size
        self._total_param_size = ((self._total_param_size + dwu_min_page_size - 1) // self._total_param_size) * self._total_param_size
        self._block_size = self._total_param_size // self._num_blocks
        self._shard_size = self._block_size // self._group_size
        if self._num_prestats > 0:
            self._prestats_shard_size = self._shard_size + self._num_prestats * self._radix_size
            self._prestats_block_size = self._prestats_shard_size * self._group_size
            self._total_param_size += self._group_size * self._num_prestats * self._radix_size

        self._flat_grads = torch.zeros([self._total_param_size]).half().cuda()
        self._new_params = None
        self._fp32_p = None
        self._fp32_m = None
        self._fp32_v = None
        self._num_redux = 0

        self._distributed_weight_update = distributed_weight_update # Is this still needed?
        self._num_rs_pg = dwu_num_rs_pg
        self._num_ar_pg = dwu_num_ar_pg
        self._num_ag_pg = dwu_num_ag_pg
        if self._num_groups > 1:
            self._ar_pg = []
            self._ar_st = []
            for dev_i in range(self._group_size):
                ranks = [dev_i+j*self._group_size for j in range(self._num_groups)]
                for i in range(self._num_ar_pg):
                    grp = torch.distributed.new_group(ranks=ranks)
                    if torch.distributed.get_rank() in ranks:
                        self._ar_pg.append(grp)
                        self._ar_st.append(torch.cuda.Stream())
        rs_ranks = []
        for group_i in range(self._num_groups):
            rs_ranks.append([group_i*self._group_size+j for j in range(self._group_size)])
        self._rs_pg = []
        self._rs_st = []
        for group_i in range(self._num_groups):
            ranks = rs_ranks[group_i]
            for i in range(self._num_rs_pg):
                grp = torch.distributed.new_group(ranks=ranks)
                if torch.distributed.get_rank() in ranks:
                    self._rs_pg.append(grp)
                    self._rs_st.append(torch.cuda.Stream())
        if self._num_ag_pg == 0:
            self._ag_pg = self._rs_pg
            self._ag_st = self._rs_st
        else:
            self._ag_pg = []
            self._ag_st = []
            for group_i in range(self._num_groups):
                ranks = rs_ranks[group_i]
                for i in range(self._num_ag_pg):
                    grp = torch.distributed.new_group(ranks=ranks)
                    if torch.distributed.get_rank() in ranks:
                        self._ag_pg.append(grp)
                        self._ag_st.append(torch.cuda.Stream())
        
    def _get_flush_block(self):
        # print([1 if x else 0 for x in self._grads_generated])
        flush_block = []
        num_grads = len(self._grads_generated)
        contiguous_idx = num_grads
        while contiguous_idx > 0 and self._grads_generated[contiguous_idx-1]:
            contiguous_idx -= 1

        if contiguous_idx < num_grads and self._grads_info[contiguous_idx]["param_offset"] <= (self._current_block-1)*self._block_size:
            self._current_block -= 1
            start = self._current_block * self._block_size
            end = (self._current_block+1) * self._block_size
            flush_block = [start, end]

        if self._current_block == 0:
            # reset
            self._grads_generated = [False]*len(self._grads_info)

        return flush_block

    def __pipeline_block_reductions(self, grad_shards):
        self._rs_st[self._num_redux%len(self._rs_st)].wait_stream(torch.cuda.current_stream())
        if self._num_groups > 1:
            with torch.cuda.stream(self._rs_st[self._num_redux%len(self._rs_st)]):
                work = torch.distributed.reduce_scatter(grad_shards[self._rank_in_group],grad_shards,group=self._rs_pg[self._num_redux%len(self._rs_pg)],async_op=True)
            with torch.cuda.stream(self._ar_st[self._num_redux%len(self._ar_st)]):
                work.wait()
                torch.distributed.all_reduce(grad_shards[self._rank_in_group],group=self._ar_pg[self._num_redux%len(self._ar_pg)],async_op=False)
            return self._ar_st[self._num_redux%len(self._ar_st)].record_event()
        else:
            with torch.cuda.stream(self._rs_st[self._num_redux%len(self._rs_st)]):
                torch.distributed.reduce_scatter(grad_shards[self._rank_in_group],grad_shards,group=self._rs_pg[self._num_redux%len(self._rs_pg)],async_op=False)
            return self._rs_st[self._num_redux%len(self._rs_st)].record_event()

    def _pipeline_block_reductions(self, flat_grads, start, end, do_prestats=False):
        if do_prestats:
            prestats_grad_block = torch.empty([self._group_size*self._prestats_shard_size]).half().cuda()
            grad_block = flat_grads[start:end]
            self._merge(prestats_grad_block, grad_block, self._decomp_stats)
            prestat_grad_shards = [prestats_grad_block[i*self._prestats_shard_size:(i+1)*self._prestats_shard_size] for i in range(self._group_size)]
            reduction_event = self.__pipeline_block_reductions(prestats_grad_shards)
            self._rs_st[self._num_redux%len(self._rs_st)].wait_event(reduction_event)
            with torch.cuda.stream(self._rs_st[self._num_redux%len(self._rs_st)]):
                grad_shards = [grad_block[i*self._shard_size:(i+1)*self._shard_size] for i in range(self._group_size)]
                self._split(prestats_grad_block, grad_shards, self._decomp_stats)
                self._extract_reduced_stats(self._decomp_stats)
            return self._rs_st[self._num_redux%len(self._rs_st)].record_event()
        else:
            grad_block = flat_grads[start:end]
            grad_shards = [grad_block[i*self._shard_size:(i+1)*self._shard_size] for i in range(self._group_size)]
            return self.__pipeline_block_reductions(grad_shards)

    # NB!
    # self._global_scale is used by this method.

    def _pipeline_block_step(self, reduction_event, flat_grads, new_params, start, end):
        grad_block = flat_grads[start:end]
        grad_shards = [grad_block[i*self._shard_size:(i+1)*self._shard_size] for i in range(self._group_size)]
        new_params_shards = [new_params[start+shard_i*self._shard_size:start+(shard_i+1)*self._shard_size] for shard_i in range(self._group_size)]
        shard_start = start + self._rank_in_group * self._shard_size
        shard_end = shard_start + self._shard_size
        self._ag_st[self._num_redux%len(self._ag_st)].wait_event(reduction_event)
        with torch.cuda.stream(self._ag_st[self._num_redux%len(self._ag_st)]):
            block_id = start // self._block_size
            self._partial_step_single_shard(block_id)
            torch.distributed.all_gather(new_params_shards,new_params_shards[self._rank_in_group],group=self._ag_pg[self._num_redux%len(self._ag_pg)],async_op=False)
        return self._ag_st[self._num_redux%len(self._ag_st)].record_event()

    def __pipeline_block_grad_L2_norm(self, reduction_event, flat_grads, start, end):
        grad_block = flat_grads[start:end]
        grad_shards = [grad_block[i*self._shard_size:(i+1)*self._shard_size] for i in range(self._group_size)]
        self._rs_st[self._num_redux%len(self._rs_st)].wait_for_event(reduction_event)
        with torch.cuda.stream(self._rs_st[self._num_redux%len(self._rs_st)]):
            self._partial_L2_grad_norm_squared[0] += fused_norm(grad_shards[self._rank_in_group])
        return self._rs_st[self._num_redux%len(self._rs_st)].record_event()

    # NB!
    # self._global_scale is set by the prestat reduction logic, OR manually by user

    def _pipeline_block(self, flat_grads, new_params, start, end, do_prestats=False):
        reduction_event = self._pipeline_block_reductions(flat_grads, start, end, do_prestats)
        if self._normalize_by_L2_grad_norm:
            reduction_event = self.__pipeline_block_grad_L2_norm(reduction_event, start, end)
        return self._pipeline_block_step(reduction_event, flat_grads, new_params, start, end)

    def _split(self, prestats_grad_block, grads_shard, prestats):
        """Split a block containing fp16 gradients and decomposed prestats.
        """
        i = self._rank_in_group
        grads_shard.copy_(prestats_grad_block[i*(self._shard_size+self._prestats_size):i*(self._shard_size+self._prestats_size)+self._shard_size])
        prestats.copy_(prestats_grad_block[i*(self._shard_size+self._prestats_size)+self._shard_size:(i+1)*(self._shard_size+self._prestats_size)])
            
    def _merge(self, prestats_grad_block, grads_block, prestats):
        """Merge fp16 gradients and decomposed prestats into single block.
        """
        for j in range(self._group_size):
            i = self._group_size - j - 1
            prestats_grad_block[i*(self._shard_size+self._prestats_size):i*(self._shard_size+self._prestats_size)+self._shard_size].copy_(grads_block[i*self._shard_size:(i+1)*self._shard_size])
            prestats_grad_block[i*(self._shard_size+self._prestats_size)+self._shard_size:(i+1)*(self._shard_size+self._prestats_size)].copy_(prestats)

    def _do_overlapped_reduction(self, param_i, param_grads_size, param_offset, grad):
        # handle overlapped reductions
        if self._last_step == False:
            torch.div(grad.view(-1), self._world_size, out=self._flat_grads[param_offset:param_offset+param_grads_size])
            self._grads_generated[param_i]=True
            flush_block = self._get_flush_block()
            while flush_block:
                start = flush_block[0]
                end = flush_block[1]
                # print("->", start, end)
                #print("block=%d :: start=%d, end=%d, fused_norm(self._flat_grads_parallel) = %e" % (self._current_block, start,end,fused_norm(self._flat_grads_parallel[start:end])))

                do_prestats = True if self._decomp_stats is not None and end == (self._block_size*self._num_blocks) else False
                if self._full_pipeline:
                    if self._new_params is None:
                        self._new_params = torch.empty_like(self._flat_grads)
                    pipe_event = self._pipeline_block(self._flat_grads, self._new_params, start, end, do_prestats)
                    self._events.append(pipe_event)
                else:
                    reduction_event = self._pipeline_block_reductions(self._flat_grads, start, end, do_prestats)
                    self._events.append(reduction_event)

                # To-Do: add events to list, wait for them before using gradient/new_params
                # To-Do: allocate self._new_params on demand

                self._num_redux += 1
                flush_block = self._get_flush_block()

            if self._current_block == 0:
                self._current_block = self.dwu_num_blocks

    def _wait_for_events(self, stream=None):
        """Wait for all events recorded during reduction and optimization pipeline.
        """
        with torch.cuda.stream(stream if stream is not None else torch.cuda.current_stream()):
            for event in self._events:
                torch.cuda.current_stream().wait_for_event(event)
        self._events = []

    def set_global_scale(self, global_scale):
        """Manually set global scale.
        Calling set_stats after this is an error and will raise a runtime error.

        """
        self._global_scale = global_scale
        self._num_prestats = 0 # clear this to prevent set_stats from being used.

    def set_stats(self, stats):
        """Provide a stats tensor that will be reduced together with the fp16 gradients.
        This method will set self._global_scale to None. 

        Arguments:
            stats: Tensor of fp32 or fp64 values that will be reduced (summed across ranks)
                   during the first block gradient reduction. The first item is used as the
                   global_scale argument to the optimizer step function.

        """
        if stats.numel() != self._num_prestats:
            raise RuntimeError('stats tensor must have %d elements' % (self._num_prestats))
        self._stats = stats
        self._decomp_stats = torch.empty([self._stats.numel()*self._radix_size]).half().cuda()
        radix_decomp_cuda.radix_decomp(self._dummy_overflow, self._stats, self._decomp_stats, self._radix_max_digit, self._radix_min_digit, self._radix_base, 1)
        self._global_scale = None
        # should we return overflow flag here or raise run time error or do nothing?

    def _extract_reduced_stats(self, decomp_stats):
        """Compose stats tensor from decomposed stats tensor after reduction.
        This method will update self._stats and self._global_scale.

        Arguments:
            decomp_stats: stats tensor radix decomposed into fp16 tensor.

        """
        radix_decomp_cuda.radix_comp(self._dummy_overflow, self._stats, decomp_stats, self._radix_max_digit, self._radix_min_digit, self._radix_base, 1)
        self._global_scale = self._stats.view([-1])[0]

    @property
    def has_overflow(self):
        """Check if overflows were detected by any call to step(...) method.
        Clears the overflow flag.
        """
        has_overflow = self._dummy_overflow_buf.item()
        self._dummy_overflow_buf.zero_()
        return has_overflow

    @property
    def peek_overflow(self):
        """Check if overflows were detected by any call to step(...) method.
        Does not clear overflow flag.
        """
        return self._dummy_overflow_buf.item()

    def strided_check_finite(self, output_params, stride=1, start=-1, end=-1, clear=True):
        """Strided check for overflow.
        You can get status by calling has_overflow.
        """
        if start >= 0 and start < end:
            out_p = output_params[start:end]
        else:
            out_p = output_params
        fused_adam_cuda.strided_check_finite(self._dummy_overflow_buf,
                out_p,
                stride,
                1 if clear else 0)

    # Distributed weight update algorithm:
    # Model parameters are kept as-is.
    # Gradients are flattened during backprop.
    # Reductions are done with an intra-node reduce-scatter followed by an inter-node all-reduce.
    # Step function is sharded and the shards are assembled with an intra-node all-gather.
    # Sharded step function needs internal fp32 buffers for p, m and v.
    # To save memory, we allocate the fp32 buffers to cover only the shards local GPU will update.
    # This means we have to play around with indexes, which requires knowledge of block and shard number.
    # Implement a method that performs a partial update of a single shard within a single block.

    def _partial_step_single_shard(self, block_id, undo=False):
        """Perform step function for a single shard.

        Arguments:
            block_id (integer): Block index of shard [0,self._num_blocks>
            undo (boolean, optional): If True, undo effect of previously called partial step.

        """
        shard_id = self._rank_in_group
        shard_start = block_id * self._block_size + shard_id * self._shard_size
        shard_end = shard_start + self._shard_size

        copy_to_fp32 = False
        if self._fp32_p is None:
            assert (not undo), "Tried to undo step before calling step."
            # Allocate fp32 buffers on demand. Note that we don't make these part of the state
            # since each rank only has partial buffers.
            # To-Do: 
            self._fp32_p = torch.empty([self._num_blocks*self._shard_size]).float().cuda()
            self._fp32_m = torch.zeros([self._num_blocks*self._shard_size]).float().cuda()
            self._fp32_v = torch.zeros([self._num_blocks*self._shard_size]).float().cuda()
            copy_to_fp32 = True

        step = None
        offset = 0
        for group in self.param_groups:
            # compute combined scale factor for this group
            combined_scale = self._global_scale
            if group['max_grad_norm'] > 0:
                # norm is in fact norm*scale
                clip = ((grad_norm / scale) + 1e-6) / group['max_grad_norm']
                if clip > 1:
                    combined_scale = clip * scale

            bias_correction = 1 if group['bias_correction'] else 0

            group_start = -1
            group_end = -2

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                if step is None:
                    # all we want from state at this point is state['step'], which should be the same for all p
                    step = state['step']
                nels = p.numel()

                start = offset
                end = start + nels
                clipped_start = start if start >= shard_start else shard_start
                clipped_end = end if end <= shard_end else shard_end
                # check if this parameter contributes to shard
                if clipped_start < clipped_end:
                    if group_start < 0:
                        group_start = clipped_start
                    group_end = clipped_end

                    if copy_to_fp32:
                        param_offset = clipped_start - shard_start
                        param_size = clipped_end - clipped_start
                        buffer_start = block_id * self._shard_size + param_offset
                        buffer_end = buffer_start + param_size
                        param_start = (clipped_start - start)
                        param_end = param_start + param_size
                        self._fp32_p[buffer_start:buffer_end].copy_(p[param_start:param_end].float())

                offset += nels

            group_size = group_end - group_start
            if group_size > 0:
                assert (step is not None), "state['step'] is None for this parameter group"
                group_offset = group_start - shard_start
                group_shard_start = shard_start + group_offset
                group_shard_end = group_shard_start + group_size
                group_buffer_start = block_id * self._shard_size + group_offset
                group_buffer_end = group_buffer_start + group_size

                beta1, beta2 = group['betas']
                if undo:
                    fused_adam_cuda.adam_undo(
                                         self._fp32_p[group_buffer_start:group_buffer_end],
                                         self._fp32_p[group_buffer_start:group_buffer_end],
                                         self._fp32_m[group_buffer_start:group_buffer_end],
                                         self._fp32_m[group_buffer_start:group_buffer_end],
                                         self._fp32_v[group_buffer_start:group_buffer_end],
                                         self._fp32_v[group_buffer_start:group_buffer_end],
                                         self._flat_grads[group_shard_start:group_shard_end],
                                         group['lr'],
                                         beta1,
                                         beta2,
                                         group['eps'],
                                         combined_scale,
                                         step+1, # To-Do: Verify this should be step+1
                                         self.eps_mode,
                                         bias_correction,
                                         group['weight_decay'])
                else:
                    fused_adam_cuda.adam(self._dummy_overflow_buf,
                                         self._fp32_p[group_buffer_start:group_buffer_end],
                                         self._fp32_p[group_buffer_start:group_buffer_end],
                                         self._new_params[group_shard_start:group_shard_end],
                                         self._fp32_m[group_buffer_start:group_buffer_end],
                                         self._fp32_m[group_buffer_start:group_buffer_end],
                                         self._fp32_v[group_buffer_start:group_buffer_end],
                                         self._fp32_v[group_buffer_start:group_buffer_end],
                                         self._flat_grads[group_shard_start:group_shard_end],
                                         group['lr'],
                                         beta1,
                                         beta2,
                                         group['eps'],
                                         combined_scale,
                                         step+1,
                                         self.eps_mode,
                                         bias_correction,
                                         group['weight_decay'])

    def revert_step(self):
        """Revert effect of previously calling partial_step.
        """
        self._wait_for_events()
        for block_id in range(self._num_blocks):
            self._partial_step_single_shard(block_id, undo=True)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self._wait_for_events()

        # Compute gradient L2 norm (optionally)
        if self._normalize_by_L2_grad_norm:
            torch.distributed.all_reduce(self._partial_L2_grad_norm_squared)
            self._L2_grad_norm = self._partial_L2_grad_norm_squared[0]

        # Complete self._new_params
        if not self._full_pipeline:
            for block_id in range(self._num_blocks):
                self._partial_step_single_shard(block_id, undo=False)

        # Check for overflow
        # Store state for loss scaler calculation
        self.strided_check_finite(self._new_params, stride=self._shard_size)
        if self.peek_overflow:
            self.revert_step()
        else:
            # Copy self._new_params to model params
            offset = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                    state['step'] += 1
                    nels = p.numel()
                    p.set_(self._new_params[offset:offset+nels].view_as(p))
                    offset += nels
            self._new_params = None

        return loss


