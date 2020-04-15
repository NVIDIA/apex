import math
import torch
import importlib
import amp_C
from apex.multi_tensor_apply import multi_tensor_applier

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
        super(DistributedFusedAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if  eps_inside_sqrt else 1

        self._overflow_buf = torch.cuda.IntTensor([0])

        assert (not flat_mt), "flat_mt option is not safe in this version"

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
        self._num_chunks = dwu_num_chunks
        self._predivide = predivide
        self._e5m2_allgather = e5m2_allgather
        self._do_not_flatten_model = do_not_flatten_model
        self._full_pipeline = full_pipeline
        self._compute_L2_grad_norm = compute_L2_grad_norm
        self._L2_grad_norm = torch.zeros([]).cuda() if self._compute_L2_grad_norm else None
        self._group_size = torch.cuda.device_count() if dwu_group_size <= 0 else dwu_group_size
        self._world_size = torch.distributed.get_world_size()
        self._num_groups = self._world_size // self._group_size
        self._rank_in_group = torch.distributed.get_rank() % self._group_size

        p_offset = 0
        p_i = 0
        self._grads_info = []
        for group in self.param_groups:
            for p in group['params']:
                torch.distributed.broadcast(p,0)
                if not p.requires_grad:
                    continue
                p_grads_size = p.numel()
                def wrapper(param, param_i, param_grads_size, param_offset):
                    def allreduce_hook(grad):
                        self._do_overlapped_reduction(param_i, param_grads_size, param_offset, grad)
                    param.register_hook(allreduce_hook)
                self._grads_info.append({"param_grads_size":p_grads_size, "param_offset":p_offset})
                wrapper(p, p_i, p_grads_size, p_offset)
                p_offset += p_grads_size
                # enforce 128b alignment (64 * fp16)
                p_offset = ((p_offset + 63) // 64) * 64 
                p_i += 1
        self._grads_generated = [False]*len(self._grads_info)
        self._flat_mt = flat_mt
        self._grads = [None]*len(self._grads_info) if self._flat_mt else None
        if self._overlap_reductions:
            self._current_block = self._num_blocks

        self._net_total_param_size = p_offset
        self._total_param_size = p_offset
        dwu_min_page_size = 256 * self._num_blocks * self._num_chunks * self._group_size
        self._total_param_size = ((self._total_param_size + dwu_min_page_size - 1) // dwu_min_page_size) * dwu_min_page_size
        self._block_size = self._total_param_size // self._num_blocks
        self._chunk_size = self._block_size // self._num_chunks
        self._shard_size = self._chunk_size // self._group_size
        print("self._net_total_param_size=%d, self._total_param_size=%d, dwu_min_page_size=%d, self._block_size=%d, self._chunk_size=%d, self._shard_size=%d" % (self._net_total_param_size, self._total_param_size,dwu_min_page_size,self._block_size,self._chunk_size,self._shard_size))

        self._low_param_i = [0]*self._num_blocks
        for block_id in range(self._num_blocks-1,-1,-1):
            p_i = len(self._grads_info)-1
            while p_i > 0 and self._grads_info[p_i]["param_offset"] > block_id*self._block_size:
                p_i -= 1
            self._low_param_i[block_id] = p_i
        print(self._low_param_i)

        self._flat_grads = torch.zeros([self._total_param_size]).half().cuda()
        self._new_params = None
        self._fp32_p = None
        self._fp32_m = None
        self._fp32_v = None
        self._copy_to_fp32 = False

        self._distributed_weight_update = distributed_weight_update # Is this still needed?
        self._num_rs_pg = dwu_num_rs_pg
        self._num_ar_pg = dwu_num_ar_pg
        self._num_ag_pg = dwu_num_ag_pg
        if self._num_groups > 1:
            self._ar_pg = []
            for dev_i in range(self._group_size):
                ranks = [dev_i+j*self._group_size for j in range(self._num_groups)]
                for i in range(self._num_ar_pg):
                    grp = torch.distributed.new_group(ranks=ranks)
                    if torch.distributed.get_rank() in ranks:
                        self._ar_pg.append(grp)
            self._ar_st = [torch.cuda.Stream()]*self._num_ar_pg
        rs_ranks = []
        for group_i in range(self._num_groups):
            rs_ranks.append([group_i*self._group_size+j for j in range(self._group_size)])
        self._rs_pg = []
        for group_i in range(self._num_groups):
            ranks = rs_ranks[group_i]
            for i in range(self._num_rs_pg):
                grp = torch.distributed.new_group(ranks=ranks)
                if torch.distributed.get_rank() in ranks:
                    self._rs_pg.append(grp)
            if self._compute_L2_grad_norm and torch.distributed.get_rank() in ranks:
                self._l2_grad_norm_pg = torch.distributed.new_group(ranks=ranks)
        self._rs_st = [torch.cuda.Stream()]*self._num_rs_pg
        if self._num_ag_pg == 0:
            self._ag_pg = self._rs_pg
            self._ag_st = self._rs_st
            self._num_ag_pg = self._num_rs_pg
        else:
            self._ag_pg = []
            for group_i in range(self._num_groups):
                ranks = rs_ranks[group_i]
                for i in range(self._num_ag_pg):
                    grp = torch.distributed.new_group(ranks=ranks)
                    if torch.distributed.get_rank() in ranks:
                        self._ag_pg.append(grp)
            self._ag_st = [torch.cuda.Stream()]*self._num_ag_pg
        self._l2_grad_norm_st = torch.cuda.Stream() if self._compute_L2_grad_norm else None
        self._completion_st = torch.cuda.Stream()

        self._reductions_works = [None]*self._num_blocks
        self._allgather_works = [None]*self._num_blocks

        import inspect
        assert ('no_copy' in inspect.getfullargspec(torch.distributed.reduce_scatter).args), "This version of c10d does not support no_copy option"


    def set_last_step(self, last_step):
        self._last_step = last_step
        
    def _get_flush_block(self):
        flush_block = []
        if self._grads_generated[self._low_param_i[self._current_block-1]]:
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

    def _pipeline_block_reductions(self, block_id):
        self._flatten_grad_mt(1.0/self._world_size if self._predivide else 1.0)

        start = block_id * self._block_size
        end = start + self._block_size
        grad_block = self._flat_grads[start:end]

        works = [None]*self._num_chunks
        for chunk in range(self._num_chunks):
            grad_chunk = grad_block[chunk*self._chunk_size:(chunk+1)*self._chunk_size]
            grad_shards = [grad_chunk[i*self._shard_size:(i+1)*self._shard_size] for i in range(self._group_size)]
            rs_stream = self._rs_st[chunk%self._num_rs_pg]
            rs_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(rs_stream):
                work = torch.distributed.reduce_scatter(grad_shards[self._rank_in_group],grad_shards,group=self._rs_pg[chunk%self._num_rs_pg],async_op=True,no_copy=True)
            if self._num_groups > 1:
                ar_stream = self._ar_st[chunk%self._num_ar_pg]
                with torch.cuda.stream(ar_stream):
                    work.wait()
                    work = torch.distributed.all_reduce(grad_shards[self._rank_in_group],group=self._ar_pg[chunk%self._num_ar_pg],async_op=True)
            works[chunk] = work

        if self._compute_L2_grad_norm:
            for chunk in range(self._num_chunks):
                grad_chunk = grad_block[chunk*self._chunk_size:(chunk+1)*self._chunk_size]
                grad_shards = [grad_chunk[i*self._shard_size:(i+1)*self._shard_size] for i in range(self._group_size)]
                with torch.cuda.stream(self._l2_grad_norm_st):
                    works[chunk].wait()
                    l2_grad_sq = grad_shards[self._rank_in_group].norm(dtype=torch.float32,p=2)**2
                    if block_id+1 == self._num_blocks and chunk == 0:
                        self._L2_grad_norm = l2_grad_sq
                    else:
                        self._L2_grad_norm += l2_grad_sq
                    if block_id == 0 and chunk+1 == self._num_chunks:
                        torch.distributed.all_reduce(self._L2_grad_norm,group=self._l2_grad_norm_pg)
                        self._L2_grad_norm.sqrt_()

        self._reductions_works[block_id] = works

    def _pipeline_block_step(self, block_id):
        if self._new_params is None:
            self._new_params = torch.zeros_like(self._flat_grads,dtype=torch.uint8 if self._e5m2_allgather else self._flat_grads.dtype)

        start = block_id * self._block_size
        end = start + self._block_size
        new_params_block = self._new_params[start:end]

        works = [None]*self._num_chunks
        for chunk in range(self._num_chunks):
            new_params_chunk = new_params_block[chunk*self._chunk_size:(chunk+1)*self._chunk_size]
            new_params_shards = [new_params_chunk[i*self._shard_size:(i+1)*self._shard_size] for i in range(self._group_size)]
            ag_stream = self._ag_st[chunk%self._num_ag_pg]
            with torch.cuda.stream(ag_stream):
                self._reductions_works[block_id][chunk].wait()
                self._partial_step_single_shard(block_id,chunk)
                work = torch.distributed.all_gather(new_params_shards,new_params_shards[self._rank_in_group],group=self._ag_pg[chunk%self._num_ag_pg],async_op=True,no_copy=True)
                works[chunk] = work

        self._allgather_works[block_id] = works

    def _flatten_grad_mt(self, scale):
        if self._flat_mt:
            grads = []
            flat_grads = []
            for p_i, (grads_info, grad) in enumerate(zip(self._grads_info, self._grads)):
                if grad is not None:
                    grads.append(grad)
                    flat_grads.append( self._flat_grads[grads_info["param_offset"]:grads_info["param_offset"]+grads_info["param_grads_size"]] )
            self._grads = [None]*len(self._grads_info)
            if len(grads) > 0:
                self._overflow_buf.zero_()
                multi_tensor_applier(
                        amp_C.multi_tensor_scale,
                        self._overflow_buf,
                        [grads, flat_grads],
                        scale)

    def _do_overlapped_reduction(self, param_i, param_grads_size, param_offset, grad):
        # handle overlapped reductions
        if self._flat_mt:
            self._grads[param_i] = grad.view(-1)
        else:
            torch.div(grad.view(-1), self._world_size if self._predivide else 1.0, out=self._flat_grads[param_offset:param_offset+param_grads_size])
        self._grads_generated[param_i]=True
        if not self._last_step:
            if self._overlap_reductions:
                flush_block = self._get_flush_block()
                while flush_block:
                    block_id = flush_block[0] // self._block_size
                    self._pipeline_block_reductions(block_id)
                    if self._full_pipeline:
                        self._pipeline_block_step(block_id)
                    flush_block = self._get_flush_block()

    def set_global_scale(self, global_scale):
        """Set global scale.
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

    def strided_check_finite(self, output_params, stride=1, start=-1, end=-1, clear=True):
        """Strided check for overflow.
        You can get status by calling has_overflow.
        """
        if start >= 0 and start < end:
            out_p = output_params[start:end]
        else:
            out_p = output_params
        fused_adam_cuda.strided_check_finite(self._overflow_buf,
                out_p,
                stride,
                1 if clear else 0)

    @property
    def L2_grad_norm(self):
        if self._compute_L2_grad_norm:
            torch.cuda.current_stream().wait_stream(self._l2_grad_norm_st)
            return self._L2_grad_norm
        else:
            return None

    # Distributed weight update algorithm:
    # Model parameters are kept as-is.
    # Gradients are flattened during backprop.
    # Reductions are done with an intra-node reduce-scatter followed by an inter-node all-reduce.
    # Step function is sharded and the shards are assembled with an intra-node all-gather.
    # Sharded step function needs internal fp32 buffers for p, m and v.
    # To save memory, we allocate the fp32 buffers to cover only the shards local GPU will update.
    # This means we have to play around with indexes, which requires knowledge of block and shard number.
    # Implement a method that performs a partial update of a single shard within a single block.

    def _partial_step_single_shard(self, block_id, chunk_id, undo=False):
        """Perform step function for a single shard.

        Arguments:
            block_id (integer): Block index of shard [0,self._num_blocks>
            undo (boolean, optional): If True, undo effect of previously called partial step.

        """
        shard_id = self._rank_in_group
        shard_start = (block_id * self._num_chunks + chunk_id) * self._chunk_size + shard_id * self._shard_size
        shard_end = shard_start + self._shard_size

        if self._fp32_p is None:
            assert (not undo), "Tried to undo step before calling step."
            # Allocate fp32 buffers on demand. Note that we don't make these part of the state
            # since each rank only has partial buffers.
            # To-Do: 
            self._fp32_p = torch.zeros([self._num_blocks*self._num_chunks*self._shard_size]).float().cuda()
            self._fp32_m = torch.zeros([self._num_blocks*self._num_chunks*self._shard_size]).float().cuda()
            self._fp32_v = torch.zeros([self._num_blocks*self._num_chunks*self._shard_size]).float().cuda()
            if self._revert_method > 1:
                self._fp32_backup_p = torch.zeros([self._num_blocks*self._num_chunks*self._shard_size]).float().cuda()
                self._fp32_backup_m = torch.zeros([self._num_blocks*self._num_chunks*self._shard_size]).float().cuda()
                self._fp32_backup_v = torch.zeros([self._num_blocks*self._num_chunks*self._shard_size]).float().cuda()
            self._copy_to_fp32 = True

        step = None
        param_i = 0
        for group in self.param_groups:
            # compute combined scale factor for this group
            combined_scale = self._global_scale
            if group['max_grad_norm'] > 0 and math.isfinite(self.L2_grad_norm):
                combined_scale = group['max_grad_norm'] / (self.L2_grad_norm / self._global_scale + 1e-6)
                combined_scale = self._global_scale / min(1, combined_scale)

            bias_correction = 1 if group['bias_correction'] else 0

            group_start = -1
            group_end = -2

            for p in group['params']:
                if not p.requires_grad:
                    continue
                #if p.grad.is_sparse:
                #    raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                if step is None:
                    # all we want from state at this point is state['step'], which should be the same for all p
                    step = state['step']
                nels = p.numel()
                offset = self._grads_info[param_i]['param_offset']
                param_i += 1

                start = offset
                end = start + nels
                clipped_start = start if start >= shard_start else shard_start
                clipped_end = end if end <= shard_end else shard_end
                # check if this parameter contributes to shard
                if clipped_start < clipped_end:
                    if group_start < 0:
                        group_start = clipped_start
                    group_end = clipped_end

                    if self._copy_to_fp32:
                        param_offset = clipped_start - shard_start
                        param_size = clipped_end - clipped_start
                        buffer_start = (block_id * self._num_chunks + chunk_id) * self._shard_size + param_offset
                        buffer_end = buffer_start + param_size
                        param_start = (clipped_start - start)
                        param_end = param_start + param_size
                        self._fp32_p[buffer_start:buffer_end].copy_(p.view(-1)[param_start:param_end].float())

            group_size = group_end - group_start
            if group_size > 0:
                assert (step is not None), "state['step'] is None for this parameter group"
                group_offset = group_start - shard_start
                group_shard_start = shard_start + group_offset
                group_shard_end = group_shard_start + group_size
                group_buffer_start = (block_id * self._num_chunks + chunk_id) * self._shard_size + group_offset
                group_buffer_end = group_buffer_start + group_size

                beta1, beta2 = group['betas']
                if undo:
                    if self._revert_method == 1:
                        fused_adam_cuda.adam_undo(
                                             self._fp32_p[group_buffer_start:group_buffer_end],
                                             self._fp32_m[group_buffer_start:group_buffer_end],
                                             self._fp32_v[group_buffer_start:group_buffer_end],
                                             self._flat_grads[group_shard_start:group_shard_end],
                                             group['lr'],
                                             beta1,
                                             beta2,
                                             group['eps'],
                                             combined_scale,
                                             step+1, # FIXME: Verify this should be step+1
                                             self.eps_mode,
                                             bias_correction,
                                             group['weight_decay'])
                    elif self._revert_method == 2:
                        self._fp32_p[group_buffer_start:group_buffer_end].copy_(self._fp32_backup_p[group_buffer_start:group_buffer_end])
                        self._fp32_m[group_buffer_start:group_buffer_end].copy_(self._fp32_backup_m[group_buffer_start:group_buffer_end])
                        self._fp32_v[group_buffer_start:group_buffer_end].copy_(self._fp32_backup_v[group_buffer_start:group_buffer_end])
                    elif self._revert_method == 3:
                        raise RuntimeError('revert_step debug option not implemented yet')
                else:
                    if self._revert_method > 1:
                        self._fp32_backup_p[group_buffer_start:group_buffer_end].copy_(self._fp32_p[group_buffer_start:group_buffer_end])
                        self._fp32_backup_m[group_buffer_start:group_buffer_end].copy_(self._fp32_m[group_buffer_start:group_buffer_end])
                        self._fp32_backup_v[group_buffer_start:group_buffer_end].copy_(self._fp32_v[group_buffer_start:group_buffer_end])
                    fused_adam_cuda.adam(
                                         self._fp32_p[group_buffer_start:group_buffer_end],
                                         self._new_params[group_shard_start:group_shard_end],
                                         self._fp32_m[group_buffer_start:group_buffer_end],
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

    def complete_reductions(self):
        """Complete reductions if full pipeline is not selected or overlap is not allowed.
        """

        if self._last_step:
            # zero out gradients that have not been completed yet
            for param_i, grad_generated in enumerate(self._grads_generated):
                if not grad_generated:
                    grad_info = self._grads_info[param_i]
                    param_offset = grad_info["param_offset"]
                    param_size = grad_info["param_grads_size"]
                    self._flat_grads[param_offset:param_offset+param_size].zero_()
                    self._grads_generated[param_i] = True

        if self._last_step or not self._overlap_reductions:
            # nothing done so far, run full pipeline after reductions
            for block_id in range(self._num_blocks-1,-1,-1):
                self._pipeline_block_reductions(block_id)

        if self._compute_L2_grad_norm:
            torch.cuda.current_stream().wait_stream(self._l2_grad_norm_st)

        self._copy_to_fp32 = False
        self._decomp_stats = None
        self._current_block = self._num_blocks
        self._grads_generated = [False]*len(self._grads_info)

    def revert_step(self):
        """Revert effect of previously calling partial_step.
        """
        for block_id in range(self._num_blocks):
            for chunk in range(self._num_chunks):
                self._partial_step_single_shard(block_id, chunk, undo=True)

    def step(self, closure=None, skip_overflow_check=False):
        loss = None
        if closure is not None:
            loss = closure()

        if self._last_step or not self._overlap_reductions or not self._full_pipeline:
            for block_id in range(self._num_blocks-1,-1,-1):
                self._pipeline_block_step(block_id)

        with torch.cuda.stream(self._completion_st):
            for block_id in range(self._num_blocks-1,-1,-1):
                for chunk in range(self._num_chunks):
                    self._allgather_works[block_id][chunk].wait()

            # Check for overflow
            # Store state for loss scaler calculation
            if skip_overflow_check:
                has_overflow = False
            else:
                self.strided_check_finite(self._new_params, stride=self._shard_size, start=0, end=self._net_total_param_size)
                has_overflow = self.peek_overflow
            if has_overflow:
                print("Reverting step")
                self.revert_step()
            else:
                # Copy self._new_params to model params
                if self._e5m2_allgather or self._do_not_flatten_model:
                    p_in = []
                    p_out = []
                with torch.no_grad():
                    param_i = 0
                    for group in self.param_groups:
                        for p in group['params']:
                            if not p.requires_grad:
                                continue
                            state = self.state[p]
                            if len(state) == 0:
                                state['step'] = 0
                            state['step'] += 1
                            nels = p.numel()
                            offset = self._grads_info[param_i]['param_offset']
                            if self._e5m2_allgather or self._do_not_flatten_model:
                                p_in.append(self._new_params[offset:offset+nels].view_as(p))
                                p_out.append(p)
                            else:
                                p.set_(self._new_params[offset:offset+nels].view_as(p))
                            param_i += 1
                    if self._e5m2_allgather:
                        multi_tensor_applier(
                                fused_adam_cuda.unpack_e5m2_mt,
                                self._overflow_buf,
                                [p_in, p_out]);
                    elif self._do_not_flatten_model:
                        multi_tensor_applier(
                                amp_C.multi_tensor_scale,
                                self._overflow_buf,
                                [p_in, p_out],
                                1.0);
            if not self._e5m2_allgather and not self._do_not_flatten_model:
                self._new_params = None

        torch.cuda.current_stream().wait_stream(self._completion_st)

        self._reductions_works = [None]*self._num_blocks
        self._allgather_works = [None]*self._num_blocks

        return loss


