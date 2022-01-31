import os
import math
import torch
import importlib
import amp_C
from apex.multi_tensor_apply import multi_tensor_applier

import torch.distributed.distributed_c10d as c10d

class DistributedFusedLAMB(torch.optim.Optimizer):

    """Implements LAMB algorithm.
    
    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.
    
    This version of fused LAMB implements 2 fusions.
      
      * Fusion of the LAMB update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.
    
    :class:`apex.optimizers.FusedLAMB`'s usage is identical to any ordinary Pytorch optimizer::
        
        opt = apex.optimizers.FusedLAMB(model.parameters(), lr = ....)
        ...
        opt.step()
    
    :class:`apex.optimizers.FusedLAMB` may be used with or without Amp.  If you wish to use :class:`FusedLAMB` with Amp,
    you may choose any ``opt_level``::
        
        opt = apex.optimizers.FusedLAMB(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()
    
    In general, ``opt_level="O1"`` is recommended.
    
    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            NOT SUPPORTED now! (default: False)
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm
            (default: 1.0)
        use_nvlamb (boolean, optional): Apply adaptive learning rate to 0.0
            weight decay parameter (default: False)
        step_supports_amp_scaling(boolean, optional): whether to use customized
            gradient unscaling logic (default: True)
    
    .. _Large Batch Optimization for Deep Learning - Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    class AtomicCounter(object):
        def __init__(self):
            self.value = 0
            self.order = []
            import threading
            self._lock = threading.Lock()

        def add(self, idx):
            with self._lock:
                self.value += 1
                self.order.append(idx)

    def __init__(self, params,
                 lr=1e-3, bias_correction = True, grad_averaging=True,
                 betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0., max_grad_norm=0., 
                 adam_w_mode=True, use_nvlamb=False,
                 step_supports_amp_scaling=True, overlap_reductions=True,
                 dwu_group_size=0, dwu_num_blocks=4, dwu_num_chunks=4,
                 dwu_num_rs_pg=1, dwu_num_ar_pg=4, dwu_num_ag_pg=0, fused_norm=False,
                 e5m2_allgather=False, verbose=False, clip_after_ar=True,
                 full_ar=False, set_param_views_to_flat_buffer=False, skip_allgather=False,
                 fuse_scale=False, param_order=None, nccl_allgather_channels=0):
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        grad_averaging=grad_averaging,
                        max_grad_norm=max_grad_norm)

        super(DistributedFusedLAMB, self).__init__(params, defaults)

        global fused_adam_cuda, distributed_lamb_cuda
        fused_adam_cuda = importlib.import_module("fused_adam_cuda")
        distributed_lamb_cuda = importlib.import_module("distributed_lamb_cuda")

        self._overflow_buf = torch.cuda.IntTensor([0])
        self._has_overflow = False
        self.multi_tensor_lamb_compute_update_term = distributed_lamb_cuda.multi_tensor_lamb_compute_update_term
        self.multi_tensor_lamb_update_weights = distributed_lamb_cuda.multi_tensor_lamb_update_weights
        import amp_C
        self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm

        self._grad_averaging = grad_averaging
        self._adam_w_mode = 1 if adam_w_mode else 0
        self._use_nvlamb = use_nvlamb
        self._step_supports_amp_scaling = step_supports_amp_scaling
        self._is_accumulation_step = False
        self._last_step = False
        self._overlap_reductions = overlap_reductions
        self._global_scale = None
        self._num_blocks = dwu_num_blocks
        self._num_chunks = dwu_num_chunks
        self._e5m2_allgather = e5m2_allgather
        self._verbose = verbose
        self._clip_after_ar = clip_after_ar
        self._full_ar = full_ar
        self._fuse_scale = fuse_scale 
        self._L2_grad_norm = None
        self._set_flat_param_view = set_param_views_to_flat_buffer
        self._skip_ag = skip_allgather
        self._fused_norm = fused_norm if not clip_after_ar else False
        self._current_process_group = c10d._get_default_group()
        self._available_ranks = list(c10d._pg_group_ranks[self._current_process_group].keys())
        self._group_size = torch.cuda.device_count() if dwu_group_size <= 0 else dwu_group_size
        self._world_size = torch.distributed.get_world_size()
        self._num_groups = self._world_size // self._group_size
        self._rank_in_group = torch.distributed.get_rank() % self._group_size

        self._lr = torch.tensor(0.0, dtype=torch.float32, device='cuda')

        self._resume_from_checkpoint = False
        self._step = torch.cuda.IntTensor([0])

        # Master weight, moment, gradient buffers
        self._fp32_p, self._fp32_m, self._fp32_v, self._fp16_p, self._fp16_g = None, None, None, None, None

        import inspect
        assert ('no_copy' in inspect.getfullargspec(torch.distributed.reduce_scatter).args), "This version of c10d does not support no_copy option"

        self._num_rs_pg = dwu_num_rs_pg
        self._num_ar_pg = dwu_num_ar_pg
        self._num_ag_pg = dwu_num_ag_pg

        if self._full_ar: # full all reduce, only need AR and AG groups
            # l2_grad_norm may be reduced within a node to limit from memory reads
            for group_i in range(self._num_groups):
                ranks = [group_i*self._group_size+j for j in range(self._group_size)]
                l2_grad_norm_pg = torch.distributed.new_group(ranks=ranks)
                if torch.distributed.get_rank() in ranks:
                    self._l2_grad_norm_pg = l2_grad_norm_pg

            self._ar_pg = []
            # consider all the ranks
            ranks = list(range(0, self._world_size))
            for i in range(self._num_ar_pg):
                if self._verbose:
                    print(f"creating new AR group {i}: {ranks}")
                grp = torch.distributed.new_group(ranks=ranks)
                if grp != torch.distributed.GroupMember.NON_GROUP_MEMBER:
                    if self._verbose:
                        print(f"group {i}: init barrier (device: {torch.cuda.current_device()})")
                    torch.distributed.barrier(group=grp, device_ids=[torch.cuda.current_device()])
                if self._verbose:
                    print(f"created new AR group {i}: {ranks}")

                if torch.distributed.get_rank() in ranks:
                    self._ar_pg.append(grp)
            self._ar_st = [torch.cuda.Stream() for _ in range(self._num_ar_pg)]
            if nccl_allgather_channels > 0:
                os.putenv('NCCL_MAX_NCHANNELS', str(nccl_allgather_channels))
            if self._num_ag_pg == 0:
                self._ag_pg = self._ar_pg
                self._ag_st = self._ar_st
                self._num_ag_pg = self._num_ar_pg
            else:
                self._ag_pg = []
                ranks = []
                stride = torch.cuda.device_count()
                for i in range(self._num_groups):
                    rs = list(range(i*stride, (i+1)*stride))
                    ranks.append(rs)
                for rs in ranks:
                    for i in range(self._num_ag_pg):
                        grp = torch.distributed.new_group(ranks=rs)
                        if torch.distributed.get_rank() in rs:
                            if self._verbose:
                                print(f"creating AG group {i}: {rs}")
                            self._ag_pg.append(grp)

                self._ag_st = [torch.cuda.Stream() for _ in range(self._num_ag_pg)]
        else: # reduce-scatter + all-reduce, need RS, AR, AG groups
            if self._num_groups > 1:
                self._ar_pg = []
                for dev_i in range(self._group_size):
                    ranks = [dev_i+j*self._group_size for j in range(self._num_groups)]
                    for i in range(self._num_ar_pg):
                        if self._verbose:
                            print(f"creating new AR group {i}: {ranks}")
                        grp = torch.distributed.new_group(ranks=ranks)
                        if grp != torch.distributed.GroupMember.NON_GROUP_MEMBER:
                            if self._verbose:
                                print(f"group {i}: init barrier (device: {torch.cuda.current_device()})")
                            torch.distributed.barrier(group=grp, device_ids=[torch.cuda.current_device()])
                        if self._verbose:
                            print(f"created new AR group {i}: {ranks}")

                        if torch.distributed.get_rank() in ranks:
                            self._ar_pg.append(grp)
                self._ar_st = [torch.cuda.Stream() for _ in range(self._num_ar_pg)]
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
                        if self._verbose:
                            print(f"creating RS group : {ranks}")
                l2_grad_norm_pg = torch.distributed.new_group(ranks=ranks)
                if torch.distributed.get_rank() in ranks:
                    self._l2_grad_norm_pg = l2_grad_norm_pg
            self._rs_st = [torch.cuda.Stream() for _ in range(self._num_rs_pg)]
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
                            if self._verbose:
                                print(f"creating AG group : {ranks}")
                self._ag_st = [torch.cuda.Stream() for _ in range(self._num_ag_pg)]
        for ag_pg in self._ag_pg:
            torch.distributed.barrier(group=ag_pg)

        self._l2_grad_norm_st = torch.cuda.Stream()
        self._completion_st = torch.cuda.Stream()
        self._step.record_stream(self._completion_st)

        self._reductions_works = [None]*self._num_blocks
        self._allgather_works = [None]*self._num_blocks

        self._one = torch.cuda.IntTensor([1])

        self._first_step = True
        self._lazy_init_stage1_done, self._lazy_init_stage2_done = False, False
        self._param_order = self.AtomicCounter()

        p_offset = 0
        p_i = 0
        self._model_params = []
        self._grad_accs = []
        self._group_properties = []
        for group in self.param_groups:
            prev = None
            beta1, beta2 = group['betas']
            beta3 = 1.0 - beta1 if self._grad_averaging else 1.0
            bias_correction = 1 if group['bias_correction'] else 0
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                self._model_params.append(p)
                self._group_properties.append((
                    weight_decay,
                    bias_correction,
                    beta1,
                    beta2,
                    beta3,
                    eps
                    ))
                p_grads_size = p.numel()
                if self._set_flat_param_view:
                    if param_order:
                        # this is executed when param_order is specified by the user
                        self._param_order.add(param_order[p])
                    else:
                        self._param_order.add(p_i)
                p_offset += p_grads_size
                # Only enforce 128b alignment (64 * fp16) for non-consecutive parameters
                # RNN is one example of consecutive parameters:
                # (weight_ih, weight_hh, bias_ih, bias_hh)
                if prev is not None and (prev.data_ptr() + prev.numel() * prev.element_size() != p.data_ptr()):
                    p_offset = ((p_offset + 63) // 64) * 64
                prev = p
                p_i += 1
        if param_order:
            self._param_order.order = torch.argsort(torch.tensor(self._param_order.order)).tolist()
        self._grads_generated = [False]*len(self._model_params)
        self._grads_fp16, self._grads_fp32 = [], []
        if self._overlap_reductions:
            self._current_block = self._num_blocks

        self._net_total_param_size = p_offset
        self._total_param_size = p_offset
        dwu_min_page_size = 256 * self._num_blocks * self._num_chunks * self._group_size
        self._total_param_size = ((self._total_param_size + dwu_min_page_size - 1) // dwu_min_page_size) * dwu_min_page_size
        self._new_params = torch.zeros([self._total_param_size], dtype=torch.uint8 if self._e5m2_allgather else torch.float16, device='cuda')



    def _lazy_init_stage1(self):
        if self._lazy_init_stage1_done: return

        p_i = 0
        #self._model_params = []
        #self._grad_accs = []
        #self._group_properties = []
        for group in self.param_groups:
            for p in group['params']:
                torch.distributed.broadcast(p, 0)
                if not p.requires_grad:
                    continue
                def wrapper(param, param_i):
                    param_tmp = param.expand_as(param)
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    def allreduce_hook(*unused):
                        if not self._set_flat_param_view:
                            if self._first_step:
                                # first time
                                self._param_order.add(param_i)
                            else:
                                idx = self._param_order.order.index(param_i)
                                self._do_overlapped_reduction(idx, param)
                        else:
                            if not self._first_step:
                                idx = self._param_order.order.index(param_i)
                                self._do_overlapped_reduction(idx, param)
                    grad_acc.register_hook(allreduce_hook)
                    self._grad_accs.append(grad_acc)
                wrapper(p, p_i)
                p_i += 1

        self._block_size = self._total_param_size // self._num_blocks
        self._chunk_size = self._block_size // self._num_chunks
        self._shard_size = self._chunk_size // self._group_size

        self._flat_grads = torch.zeros([self._total_param_size], dtype=torch.float16, device='cuda')
        self._mega_shard_size = self._num_blocks * self._num_chunks * self._shard_size
        # initialize master weights, moments buffers if not loaded from checkpoint
        if self._fp32_p is None:
            self._fp32_p = torch.zeros([self._mega_shard_size], dtype=torch.float32, device='cuda')
            self._fp32_m = torch.zeros([self._mega_shard_size], dtype=torch.float32, device='cuda')
            self._fp32_v = torch.zeros([self._mega_shard_size], dtype=torch.float32, device='cuda')
            self._fp32_u = torch.zeros([self._mega_shard_size], dtype=torch.float32, device='cuda')
        # FIXME: Rethink fp16 label since it's either uint8 or fp16
        self._fp16_p = torch.zeros([self._mega_shard_size], dtype=torch.uint8 if self._e5m2_allgather else torch.float16, device='cuda')
        self._fp16_g = torch.zeros([self._mega_shard_size], dtype=torch.float16, device='cuda')

        def _flat_split(p):
            def __blockify(p):
                return [p[block_id*self._block_size:(block_id+1)*self._block_size] for block_id in range(self._num_blocks)]
            def __chunkify(p):
                return [p[chunk_id*self._chunk_size:(chunk_id+1)*self._chunk_size] for chunk_id in range(self._num_chunks)]
            def __shardify(p):
                return [p[shard_id*self._shard_size:(shard_id+1)*self._shard_size] for shard_id in range(self._group_size)]
            list_of_blocks = __blockify(p)
            list_of_list_of_chunks = [__chunkify(block) for block in list_of_blocks]
            list_of_list_of_list_of_shards = [[__shardify(chunk) for chunk in chunks] for chunks in list_of_list_of_chunks]
            return list_of_blocks, list_of_list_of_chunks, list_of_list_of_list_of_shards
        def _flat_split_no_shards(p):
            def __blockify(p):
                return [p[block_id*self._block_size:(block_id+1)*self._block_size] for block_id in range(self._num_blocks)]
            def __chunkify(p):
                return [p[chunk_id*self._chunk_size:(chunk_id+1)*self._chunk_size] for chunk_id in range(self._num_chunks)]
            list_of_blocks = __blockify(self._flat_grads)
            list_of_list_of_chunks = [__chunkify(block) for block in list_of_blocks]
            return list_of_blocks, list_of_list_of_chunks
        def _full_packed_split(p):
            def __shardify(p):
                return [p[mega_shard*self._mega_shard_size:(mega_shard+1)*self._mega_shard_size] for mega_shard in range(self._group_size)]
            def __blockify(p):
                return [p[block_id*self._num_chunks*self._shard_size:(block_id+1)*self._num_chunks*self._shard_size] for block_id in range(self._num_blocks)]
            def __chunkify(p):
                return [p[chunk_id*self._shard_size:(chunk_id+1)*self._shard_size] for chunk_id in range(self._num_chunks)]
            list_of_mega_shards = __shardify(p)
            list_of_list_of_mega_blocks = [__blockify(mega_shard) for mega_shard in list_of_mega_shards]
            list_of_list_of_list_of_mega_chunks = [[__chunkify(mega_block) for mega_block in mega_blocks] for mega_blocks in list_of_list_of_mega_blocks]
            return list_of_mega_shards, list_of_list_of_mega_blocks, list_of_list_of_list_of_mega_chunks
        def _packed_split(p):
            def __packed_blockify(p):
                packed_block_size = self._num_chunks*self._shard_size
                return [p[block_id*packed_block_size:(block_id+1)*packed_block_size] for block_id in range(self._num_blocks)]
            def __packed_chunkify(p):
                # in the packed format, each chunk contains one shard, so packed_chunk_size == self._shard_size
                return [p[chunk_id*self._shard_size:(chunk_id+1)*self._shard_size] for chunk_id in range(self._num_chunks)]
            list_of_blocks = __packed_blockify(p)
            list_of_list_of_chunks = [__packed_chunkify(block) for block in list_of_blocks]
            return list_of_blocks, list_of_list_of_chunks
        def _split_assign(shards):
            packed_block_size = self._num_chunks*self._shard_size
            list_of_list_of_chunks=[]
            for block_id in range(self._num_blocks):
                list_of_chunks=[]
                for chunk_id in range(self._num_chunks):
                    #self._fp16_g[block_id*packed_block_size+chunk_id*self._shard_size:block_id*packed_block_size+(chunk_id+1)*self._shard_size] = shards[block_id][chunk_id][self._rank_in_group]
                    list_of_chunks.append( shards[block_id][chunk_id][self._rank_in_group])
                list_of_list_of_chunks.append(list_of_chunks)
            return list_of_list_of_chunks

        self._new_params_mega_shards, self._new_params_mega_blocks, self._new_params_mega_chunks = _full_packed_split(self._new_params)
        # this splitting scheme is needed when allgather needs to be split into multiple chunks in a contiguous way
        self._new_params2_blocks, self._new_params2_chunks, self._new_params2_shards = _flat_split(self._new_params)

        self._fp32_p_blocks, self._fp32_p_chunks = _packed_split(self._fp32_p)
        self._fp32_m_blocks, self._fp32_m_chunks = _packed_split(self._fp32_m)
        self._fp32_v_blocks, self._fp32_v_chunks = _packed_split(self._fp32_v)
        self._fp32_u_blocks, self._fp32_u_chunks = _packed_split(self._fp32_u)
        self._fp16_p_blocks, self._fp16_p_chunks = _packed_split(self._fp16_p)

        if self._full_ar:
            # for gradient all-reduce
            self._flat_grads_blocks, self._flat_grads_chunks, self._flat_grads_shards = _flat_split(self._flat_grads)
            # for weight update
            self._fp16_g_chunks = _split_assign(self._flat_grads_shards)
        else:
            self._flat_grads_blocks, self._flat_grads_chunks, self._flat_grads_shards = _flat_split(self._flat_grads)
            self._fp16_g_blocks, self._fp16_g_chunks = _packed_split(self._fp16_g)

        self._lazy_init_stage1_done = True

    def _lazy_init_stage2(self):
        if self._lazy_init_stage2_done: return
        if not self._set_flat_param_view: 
            # reversing is needed for overlapping allreduce and backprop, but currently not supported for flat param view
            self._param_order.order.reverse()

            # re-order model_params, grad_accs, group_properties lists
        self._model_params = [self._model_params[i] for i in self._param_order.order]
        self._grad_accs = [self._grad_accs[i] for i in self._param_order.order]
        self._group_properties = [self._group_properties[i] for i in self._param_order.order]

        def _get_flat_view(param):
            if param.is_contiguous(memory_format=torch.channels_last):
                K, C, H, W = param.shape
                pv = param.as_strided(size=(K,H,W,C), stride=(H*W*C, W*C, C, 1))
            elif param.is_contiguous(memory_format=torch.channels_last_3d):
                K, C, D, H, W = param.shape
                pv = param.as_strided(size=(K,D,H,W,C), stride=(D*H*W*C, H*W*C, W*C, C, 1))
            else:
                pv = param
            return pv.view(-1)

        # re-collect grads info (size, offset) after ordering
        prev = None
        p_offset = 0
        self._grads_info = []
        self._individual_flat_grads = []
        for i, p in enumerate(self._model_params):
            p_grads_size = p.numel()
            self._grads_info.append({"param_grads_size":p_grads_size, "param_offset":p_offset})
            self._individual_flat_grads.append(self._flat_grads[p_offset:p_offset+p_grads_size].view_as(p))
            # for the first iteration
            self._do_overlapped_reduction(i, p)
            p_offset += p_grads_size
            # Only enforce 128b alignment (64 * fp16) for non-consecutive parameters
            # RNN is one example of consecutive parameters:
            # (weight_ih, weight_hh, bias_ih, bias_hh)
            if prev is not None and (prev.data_ptr() + prev.numel() * prev.element_size() != p.data_ptr()):
                p_offset = ((p_offset + 63) // 64) * 64
            prev = p

        self._low_param_i = [0]*self._num_blocks
        for block_id in range(self._num_blocks-1,-1,-1):
            p_i = len(self._grads_info)-1
            while p_i > 0 and self._grads_info[p_i]["param_offset"] > block_id*self._block_size:
                p_i -= 1
            self._low_param_i[block_id] = p_i
        #print("self._low_param_i", self._low_param_i)

        # This paragraph does two things:
        # 1) Copy model parameters into master buffer
        # 2) Create tensor lists for unpacking new parameter tensor after all-gather
        self._packed_flat_to_model_params_fp16 = []
        self._packed_flat_to_model_params_fp32 = []
        self._model_params_num = len(self._model_params)
        self._contrib_tensor_list = []
        self._contrib_min_param_i, self._contrib_max_param_i = -1, -1
        self._contrib_update_frag_for_norm = []
        self._contrib_model_param_for_norm_fp16 = []
        self._contrib_model_param_for_norm_fp32 = []
        self._contrib_model_param_for_norm_is_fp16 = []
        self._model_param_is_contrib = []
        self._contrib_group_properties = []
        for shard_id in range(self._group_size):
            for block_id in range(self._num_blocks):
                for chunk_id in range(self._num_chunks):
                    flat_shard_start = (((block_id * self._num_chunks + chunk_id) * self._group_size) + shard_id) * self._shard_size
                    flat_shard_end = flat_shard_start + self._shard_size
                    for param_i, (p, grads_info, group_props) in enumerate(zip(self._model_params, self._grads_info, self._group_properties)):
                        flat_grad_start = grads_info["param_offset"]
                        flat_grad_end = flat_grad_start + grads_info["param_grads_size"]
                        clipped_start = (lambda a,b: a if a > b else b)(flat_grad_start, flat_shard_start)
                        clipped_end = (lambda a,b: a if a < b else b)(flat_grad_end, flat_shard_end)
                        if clipped_start < clipped_end:
                            grad_offset = clipped_start - flat_grad_start
                            grad_length = clipped_end - clipped_start
                            shard_offset = clipped_start - flat_shard_start
                            pf = _get_flat_view(p)
                            model_param_fragment = pf[grad_offset:grad_offset+grad_length]
                            new_param_packed_fragment = self._new_params_mega_chunks[shard_id][block_id][chunk_id][shard_offset:shard_offset+grad_length]
                            if model_param_fragment.dtype == torch.float16:
                                self._packed_flat_to_model_params_fp16.append( (new_param_packed_fragment, model_param_fragment) )
                            else:
                                self._packed_flat_to_model_params_fp32.append( (new_param_packed_fragment, model_param_fragment) )
                            if shard_id == self._rank_in_group:
                                self._model_param_is_contrib.append(param_i)
                                # copy model parameters into master buffer
                                master_param_fragment = self._fp32_p_chunks[block_id][chunk_id][shard_offset:shard_offset+grad_length]
                                opti_state_m_fragment = self._fp32_m_chunks[block_id][chunk_id][shard_offset:shard_offset+grad_length]
                                opti_state_v_fragment = self._fp32_v_chunks[block_id][chunk_id][shard_offset:shard_offset+grad_length]
                                opti_state_u_fragment = self._fp32_u_chunks[block_id][chunk_id][shard_offset:shard_offset+grad_length]
                                opti_state_g_fragment = self._fp16_g_chunks[block_id][chunk_id][shard_offset:shard_offset+grad_length]
                                opti_state_p_fragment = self._fp16_p_chunks[block_id][chunk_id][shard_offset:shard_offset+grad_length]
                                #print("model_param_fragment.size()=%s, new_param_packed_fragment.size()=%s, master_param_fragment.size()=%s" % (str(model_param_fragment.size()), str(new_param_packed_fragment.size()), str(master_param_fragment.size())))
                                if not self._resume_from_checkpoint:
                                    master_param_fragment.copy_(model_param_fragment)
                                self._contrib_group_properties.append(group_props)
                                self._contrib_tensor_list.append((master_param_fragment, opti_state_m_fragment, opti_state_v_fragment, opti_state_u_fragment, opti_state_g_fragment, opti_state_p_fragment)) # p, m, v, u, g, p_copy
                                self._contrib_update_frag_for_norm.append(opti_state_u_fragment)
                                if p.dtype == torch.float16:
                                    self._contrib_model_param_for_norm_fp16.append(p)
                                else:
                                    self._contrib_model_param_for_norm_fp32.append(p)
                                self._contrib_model_param_for_norm_is_fp16.append(True if p.dtype == torch.float16 else False)
                                if self._contrib_min_param_i < 0: self._contrib_min_param_i = param_i
                                self._contrib_max_param_i = param_i
        self._contrib_model_param_for_norm_num = len(self._contrib_model_param_for_norm_is_fp16)
        if len(self._contrib_model_param_for_norm_fp16) == 0: self._contrib_model_param_for_norm_fp16 = None
        if len(self._contrib_model_param_for_norm_fp32) == 0: self._contrib_model_param_for_norm_fp32 = None
        self._contrib_model_param_for_norm_is_fp32 = torch.tensor([not is_fp16 for is_fp16 in self._contrib_model_param_for_norm_is_fp16], dtype=torch.bool, device='cuda')
        self._contrib_model_param_for_norm_is_fp16 = torch.tensor([is_fp16 for is_fp16 in self._contrib_model_param_for_norm_is_fp16], dtype=torch.bool, device='cuda')
        self._offsets = torch.tensor(self._model_param_is_contrib, dtype=torch.int64, device='cuda')

        p, m, v, u, g, p_copy = list(zip(*self._contrib_tensor_list))
        self._contrib_compute_update_term_tensor_list = [g, p, m, v, u]
        self._contrib_update_weights_tensor_list = [u, p, p_copy]

        math_type = self._fp32_u.dtype
        decay, bias_correction, beta1, beta2, beta3, epsilon = list(zip(*self._contrib_group_properties))
        self._contrib_beta1 = torch.tensor(beta1, dtype=math_type, device='cuda')
        self._contrib_beta2 = torch.tensor(beta2, dtype=math_type, device='cuda')
        self._contrib_beta3 = torch.tensor(beta3, dtype=math_type, device='cuda')
        self._contrib_bias_correction = torch.tensor(bias_correction, dtype=torch.int, device='cuda')
        self._contrib_epsilon = torch.tensor(epsilon, dtype=math_type, device='cuda')
        self._contrib_weight_decay = torch.tensor(decay, dtype=math_type, device='cuda')

        self._packed_flat_to_model_params_fp16 = list(zip(*self._packed_flat_to_model_params_fp16)) if len(self._packed_flat_to_model_params_fp16) > 0 else None
        self._packed_flat_to_model_params_fp32 = list(zip(*self._packed_flat_to_model_params_fp32)) if len(self._packed_flat_to_model_params_fp32) > 0 else None

        self._lazy_init_stage2_done = True

        self.complete_reductions()
        self._first_step = False

    def set_is_accumulation_step(self, is_accumulation_step):
        self._is_accumulation_step = is_accumulation_step

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

    def _full_all_reduce_scale(self, block_id, scale):
        works = [None]*self._num_chunks
        if  self._clip_after_ar:    
            for chunk_id in range(self._num_chunks):
                glob_chunk_id = block_id * self._num_chunks + chunk_id
                ar_stream = self._ar_st[glob_chunk_id%self._num_ar_pg]
                ar_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(ar_stream):
                    works[chunk_id] = torch.distributed.all_reduce(self._flat_grads_chunks[block_id][chunk_id],group=self._ar_pg[glob_chunk_id%self._num_ar_pg],async_op=True,op=torch.distributed.make_nccl_premul_sum((scale,)))
        else:
            glob_chunk_id = block_id
            ar_stream = self._ar_st[glob_chunk_id%self._num_ar_pg]
            ar_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ar_stream):
                    works0 = torch.distributed.all_reduce(self._flat_grads_blocks[block_id],group=self._ar_pg[glob_chunk_id%self._num_ar_pg],async_op=True,op=torch.distributed.make_nccl_premul_sum((scale,)))
            for i in range(self._num_chunks):
                works[i]=works0
        self._reductions_works[block_id] = works

    def _full_all_reduce(self, block_id):
        works = [None]*self._num_chunks

        for chunk_id in range(self._num_chunks):
            glob_chunk_id = block_id * self._num_chunks + chunk_id
            ar_stream = self._ar_st[glob_chunk_id%self._num_ar_pg]
            ar_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ar_stream):
                works[chunk_id] = torch.distributed.all_reduce(self._flat_grads_chunks[block_id][chunk_id],group=self._ar_pg[glob_chunk_id%self._num_ar_pg],async_op=True)
        self._reductions_works[block_id] = works

    def _reduce_scatter_and_all_reduce_scale(self, block_id, scale):
        # Reduction within each node
        # Changes gradient format from [block * chunk * shard] to [shard * block * chunk]
        # The output format is the same as the fp32 master parameters
        works = [None]*self._num_chunks
        for chunk_id in range(self._num_chunks):
            glob_chunk_id = block_id * self._num_chunks + chunk_id
            rs_stream = self._rs_st[glob_chunk_id%self._num_rs_pg]
            rs_stream.wait_stream(torch.cuda.current_stream())
            rs_stream.wait_stream(self._l2_grad_norm_st)
            with torch.cuda.stream(rs_stream):
                works[chunk_id] = torch.distributed.reduce_scatter(self._fp16_g_chunks[block_id][chunk_id],self._flat_grads_shards[block_id][chunk_id],group=self._rs_pg[glob_chunk_id%self._num_rs_pg],async_op=True,no_copy=True,op=torch.distributed.make_nccl_premul_sum((scale,)))

        # Reduction across nodes for each rank
        if self._num_groups > 1:
            for chunk_id in range(self._num_chunks):
                glob_chunk_id = block_id * self._num_chunks + chunk_id
                ar_stream = self._ar_st[glob_chunk_id%self._num_ar_pg]
                with torch.cuda.stream(ar_stream):
                    works[chunk_id].wait()
                    works[chunk_id] = torch.distributed.all_reduce(self._fp16_g_chunks[block_id][chunk_id],group=self._ar_pg[glob_chunk_id%self._num_ar_pg],async_op=True)
        self._reductions_works[block_id] = works

    def _reduce_scatter_and_all_reduce(self, block_id):
        # Reduction within each node
        # Changes gradient format from [block * chunk * shard] to [shard * block * chunk]
        # The output format is the same as the fp32 master parameters
        works = [None]*self._num_chunks
        for chunk_id in range(self._num_chunks):
            glob_chunk_id = block_id * self._num_chunks + chunk_id
            rs_stream = self._rs_st[glob_chunk_id%self._num_rs_pg]
            rs_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(rs_stream):
                works[chunk_id] = torch.distributed.reduce_scatter(self._fp16_g_chunks[block_id][chunk_id],self._flat_grads_shards[block_id][chunk_id],group=self._rs_pg[glob_chunk_id%self._num_rs_pg],async_op=True,no_copy=True)

        # Reduction across nodes for each rank
        if self._num_groups > 1:
            for chunk_id in range(self._num_chunks):
                glob_chunk_id = block_id * self._num_chunks + chunk_id
                ar_stream = self._ar_st[glob_chunk_id%self._num_ar_pg]
                with torch.cuda.stream(ar_stream):
                    works[chunk_id].wait()
                    works[chunk_id] = torch.distributed.all_reduce(self._fp16_g_chunks[block_id][chunk_id],group=self._ar_pg[glob_chunk_id%self._num_ar_pg],async_op=True)
        self._reductions_works[block_id] = works

    def _pipeline_block_reductions(self, block_id):
        if self._clip_after_ar:
            self._flatten_grad_mt(1.0/self._world_size)

            if self._full_ar:
                self._full_all_reduce(block_id)
            else:
                self._reduce_scatter_and_all_reduce(block_id)

            # Compute L2 grad norm
            if block_id == 0:
                with torch.cuda.stream(self._l2_grad_norm_st):
                    for block_id in range(self._num_blocks):
                        for chunk_id in range(self._num_chunks):
                            self._reductions_works[block_id][chunk_id].wait()
                    # Since the packed format is contiguous after reductions, only one norm is needed
                    l2_grad_norm_sq = torch.empty([1], device='cuda')
                    if self._full_ar:
                        # this flattening of lists is to keep multi_tensor_apply function happy, it wants depth=1 for l2 norm computation
                        flat_list = [item for sublist in self._fp16_g_chunks for item in sublist]
                        l2_grad_norm_sq = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [flat_list], False)[0]**2
                    else:
                        l2_grad_norm_sq = self._fp16_g.norm(dtype=torch.float32, p=2)**2
                    torch.distributed.all_reduce(l2_grad_norm_sq, group=self._l2_grad_norm_pg)
                    self._L2_grad_norm = l2_grad_norm_sq.sqrt()
        else:
            # Copy model grads to flat grads buffer
            self._flatten_grad_mt(1.0)

            # Compute L2 grad norm
            self._l2_grad_norm_st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._l2_grad_norm_st):
                if not self._fused_norm:
                    self._L2_grad_norm = self._flat_grads.norm(dtype=torch.float16, p=2).float()
            torch.cuda.current_stream().wait_stream(self._l2_grad_norm_st)

            # Apply clipping & pre-reduction scaling on grads
            loss_scale = self.global_scale
            max_grad_norm = loss_scale*self.defaults['max_grad_norm']
            coeff = max_grad_norm /(1e-6+self.L2_grad_norm)
            coeff = (coeff>1) * self._one + (coeff<=1) * coeff
            tmp = torch.cat(((self._one), (coeff)))
            index = (coeff+1>coeff).int()
            scale = tmp.index_select(0, index).half()/self._world_size
            if not self._fuse_scale:
                self._flat_grads.mul_(scale)

            if self._full_ar:
                if self._fuse_scale:
                    self._full_all_reduce_scale(block_id, scale)
                else:
                    self._full_all_reduce(block_id) 
            else:
                if self._fuse_scale:
                    self._reduce_scatter_and_all_reduce_scale(block_id, scale)
                else:
                    self._reduce_scatter_and_all_reduce(block_id)

            if block_id == 0:
                for block_id in range(self._num_blocks):
                    for chunk_id in range(self._num_chunks):
                        self._reductions_works[block_id][chunk_id].wait()

    def __compute_contrib_param_norm(self):
        if self._contrib_model_param_for_norm_fp16 is not None and self._contrib_model_param_for_norm_fp32 is not None:
            gnorm_fp16 = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [self._contrib_model_param_for_norm_fp16], True)[1]
            gnorm_fp32 = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [self._contrib_model_param_for_norm_fp32], True)[1]
            gnorm = torch.empty(size=[self._contrib_model_param_for_norm_num], dtype=torch.bool, device='cuda')
            gnorm.masked_scatter_(self._contrib_model_param_for_norm_is_fp16, gnorm_fp16)
            gnorm.masked_scatter_(self._contrib_model_param_for_norm_is_fp32, gnorm_fp32)
        elif self._contrib_model_param_for_norm_fp16 is not None:
            gnorm = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [self._contrib_model_param_for_norm_fp16], True)[1]
        elif self._contrib_model_param_for_norm_fp32 is not None:
            gnorm = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [self._contrib_model_param_for_norm_fp32], True)[1]
        return gnorm

    def __compute_contrib_update_norm(self):
        l2_norm = torch.zeros(size=[self._model_params_num], dtype=torch.float32, device='cuda')
        local_contrib_l2_norm = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [self._contrib_update_frag_for_norm], True)[1] ** 2
        l2_norm.scatter_(dim=0, index=self._offsets, src=local_contrib_l2_norm)
        torch.distributed.all_reduce(l2_norm, group=self._ag_pg[0])
        l2_norm = torch.sqrt(l2_norm)
        return l2_norm

    def _pipeline_step(self):
        global_scale = self.global_scale
        # if clip before ar, set max_grad_norm to 0
        max_grad_norm = self.defaults['max_grad_norm'] * self._clip_after_ar
        self._completion_st.wait_stream(self._l2_grad_norm_st)
        global_grad_norm = self.L2_grad_norm

        # check global_grad_norm and fill overflow_buf
        is_finite = (global_grad_norm + 1 > global_grad_norm).int()
        self._overflow_buf = self._one * (is_finite ^ self._one) # toggle between 0 and 1

        if not self._clip_after_ar:
            torch.distributed.all_reduce(is_finite,
                                         op=torch.distributed.ReduceOp.MIN,
                                         group=self._current_process_group)
            torch.distributed.all_reduce(self._overflow_buf,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=self._current_process_group)

        # increment step counter if no overflow
        self._step += is_finite
        self._completion_st.wait_stream(torch.cuda.current_stream())
        self._completion_st.wait_stream(self._l2_grad_norm_st)

        # Call step kernel once per step
        # Call all-gather once per step
        with torch.cuda.stream(self._completion_st):
            for block_id in range(self._num_blocks):
                for chunk_id in range(self._num_chunks):
                    self._reductions_works[block_id][chunk_id].wait()
            param_norm = self.__compute_contrib_param_norm()
            multi_tensor_applier(self.multi_tensor_lamb_compute_update_term,
                    self._overflow_buf,
                    self._contrib_compute_update_term_tensor_list, # g, p, m, v, u
                    self._contrib_beta1,
                    self._contrib_beta2,
                    self._contrib_beta3,
                    self._contrib_bias_correction,
                    self._step,
                    self._contrib_epsilon,
                    self._adam_w_mode,
                    self._contrib_weight_decay,
                    global_scale,
                    global_grad_norm,
                    max_grad_norm)
            upd_norm = self.__compute_contrib_update_norm()
            multi_tensor_applier(self.multi_tensor_lamb_update_weights,
                    self._overflow_buf,
                    self._contrib_update_weights_tensor_list, # u, p, p_copy
                    param_norm,
                    upd_norm,
                    self._offsets,
                    self._lr,
                    self._contrib_weight_decay,
                    global_grad_norm,
                    self._use_nvlamb)
            if not self._skip_ag:
                # allgather chunking is currently not supported for clip after allreduce 
                if not self._clip_after_ar:
                    for block in range(self._num_blocks):
                        for chunk in range(self._num_chunks):
                            torch.distributed.all_gather(self._new_params2_shards[block][chunk], self._fp16_p_chunks[block][chunk], group=self._ag_pg[0], no_copy=True)
                else:
                    torch.distributed.all_gather(self._new_params_mega_shards, self._fp16_p, group=self._ag_pg[0], no_copy=True)

    def _flatten_grad_mt(self, scale):
        if len(self._grads_fp16) > 0:
            self._overflow_buf.zero_()
            if not self._fused_norm:
                multi_tensor_applier(
                        amp_C.multi_tensor_scale,
                        self._overflow_buf,
                        list(zip(*self._grads_fp16)),
                        scale)
            else:
                self._L2_grad_norm=multi_tensor_applier(
                        amp_C.multi_tensor_l2norm_scale,
                        self._overflow_buf,
                        list(zip(*self._grads_fp16)),
                        scale, False)[0].float()

            self._grads_fp16 = []
        if len(self._grads_fp32) > 0:
            self._overflow_buf.zero_()
            if not self._fused_norm:
                multi_tensor_applier(
                        amp_C.multi_tensor_scale,
                        self._overflow_buf,
                        list(zip(*self._grads_fp32)),
                        scale)
            else:
                self._L2_grad_norm=multi_tensor_applier(
                        amp_C.multi_tensor_l2norm_scale,
                        self._overflow_buf,
                        list(zip(*self._grads_fp32)),
                        scale, False)[0].float()
            self._grads_fp32 = []

    def _do_overlapped_reduction(self, param_i, param):
        if not self._is_accumulation_step:
            # handle overlapped reductions
            if param.dtype == torch.float16:
                self._grads_fp16.append( (param.grad, self._individual_flat_grads[param_i]) )
            else:
                self._grads_fp32.append( (param.grad, self._individual_flat_grads[param_i]) )
            self._grads_generated[param_i]=True
            if not self._first_step and not self._last_step:
                if self._overlap_reductions:
                    flush_block = self._get_flush_block()
                    while flush_block:
                        block_id = flush_block[0] // self._block_size
                        self._pipeline_block_reductions(block_id)
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
            for param_i, grad_generated in enumerate(self._grads_generated):
                if not grad_generated:
                    grad_info = self._grads_info[param_i]
                    param_offset = grad_info["param_offset"]
                    param_size = grad_info["param_grads_size"]
                    self._flat_grads[param_offset:param_offset+param_size].zero_()
                    self._grads_generated[param_i] = True

        if self._first_step or self._last_step or not self._overlap_reductions:
            # nothing done so far, run full pipeline after reductions
            for block_id in range(self._num_blocks-1,-1,-1):
                self._pipeline_block_reductions(block_id)

        torch.cuda.current_stream().wait_stream(self._l2_grad_norm_st)

        self._current_block = self._num_blocks
        self._grads_generated = [False]*len(self._grads_info)

    def step(self, closure=None, grad_scaler=None):
        loss = None
        if closure is not None:
            loss = closure()

        self._pipeline_step()

        if grad_scaler is not None:
            found_inf = self._overflow_buf.float()
            optimizer_state = grad_scaler._per_optimizer_states[id(self)]
            current_device = torch.device('cuda', torch.cuda.current_device())
            optimizer_state["found_inf_per_device"][current_device] = found_inf

        self._completion_st.wait_stream(torch.cuda.current_stream())
        if not self._set_flat_param_view:
            with torch.cuda.stream(self._completion_st):
                # Copy self._new_params to model params
                with torch.no_grad():
                    if self._packed_flat_to_model_params_fp16 is not None:
                        multi_tensor_applier(
                                fused_adam_cuda.maybe_cast_mt,
                                self._overflow_buf,
                                self._packed_flat_to_model_params_fp16)
                    if self._packed_flat_to_model_params_fp32 is not None:
                        multi_tensor_applier(
                                fused_adam_cuda.maybe_cast_mt,
                                self._overflow_buf,
                                self._packed_flat_to_model_params_fp32)
    
        torch.cuda.current_stream().wait_stream(self._completion_st)

        self._reductions_works = [None]*self._num_blocks
        self._allgather_works = [None]*self._num_blocks

        return loss

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`DistributedFusedAdam` instance.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        # save step, master weights and first/second moments
        state_dict = {}
        state_dict['step'] = self._step
        state_dict['fp32_p'] = self._fp32_p
        state_dict['fp32_m'] = self._fp32_m
        state_dict['fp32_v'] = self._fp32_v
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If an DistributedFusedAdam instance was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``optimizer.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        # restore step, master weights and first/second moments
        self._step = state_dict['step']
        self._fp32_p = state_dict['fp32_p'].to(device="cuda")
        self._fp32_m = state_dict['fp32_m'].to(device="cuda")
        self._fp32_v = state_dict['fp32_v'].to(device="cuda")
        self._resume_from_checkpoint = True
