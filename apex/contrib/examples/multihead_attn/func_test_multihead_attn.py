import torch
import torch.nn.functional as F
import argparse

from apex.contrib.multihead_attn import SelfMultiheadAttn
from apex.contrib.multihead_attn import EncdecMultiheadAttn

parser = argparse.ArgumentParser(description='Multihead Attention Standalone Test')
parser.add_argument('--seq-length', default=64, type=int, help='Sequence Length of Input')
parser.add_argument('--num-seqs-start', default=5, type=int, help='Start Range of Number of Sequences')
parser.add_argument('--num-seqs-stop', default=80, type=int, help='Stop Range of Number of Sequences')
parser.add_argument('--num-seqs-inc', default=5, type=int, help='Range Increment of Number of Sequences')
parser.add_argument('--trials', default=20, type=int, help='Number of Trials to Execute')
parser.add_argument('--warmup-trials', default=5, type=int, help='Warmup Trials to discard')
parser.add_argument('--layers', default=18, type=int, help='Attention Layers to Execute to Gain CPU/GPU Time Overlap')
parser.add_argument('--seed-start', default=1, type=int, help='Attention Layers to Execute to Gain CPU/GPU Time Overlap')
parser.add_argument('--seed-end', default=100, type=int, help='Attention Layers to Execute to Gain CPU/GPU Time Overlap')
parser.add_argument('--hidden-dim', default=1024, type=int, help='Multihead Attention hidden dimension')
parser.add_argument('--heads', default=16, type=int, help='Number of Multihead Attention heads')
parser.add_argument('--encdec-attn', action='store_true', help='Use Encoder-Decoder Attention instead of Self Attention.')
parser.add_argument('--norm-add', action='store_true', help='Include Layer Norm and Dropout-Add in Multihead Attention block.')
parser.add_argument('--ref', action='store_true', help='Reference implementation in python pytorch.')
parser.add_argument('--native', action='store_true', help='torch.nn.MultitheadAttention Version.')
parser.add_argument('--fwd', action='store_true', help='Only execute Fwd Pass.')
parser.add_argument('--eval', action='store_true', help='Inference only, no backward pass.')

args = parser.parse_args()
assert args.seq_length % 64 == 0, "Sequence Length should be a multiple of 64!"

if not torch.cuda.is_available():
    raise NotImplementedError('Running on CPU is not supported')
torch.cuda.set_device(0)

dropout_prob = 0.1

for seed in range(args.seed_start, args.seed_end+1) :
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ref_layer = None
    if args.encdec_attn :
        ref_layer = EncdecMultiheadAttn(args.hidden_dim, args.heads, dropout=dropout_prob, bias=False, include_norm_add=args.norm_add, impl='default')
    else :
        ref_layer = SelfMultiheadAttn(args.hidden_dim, args.heads, dropout=dropout_prob, bias=False, include_norm_add=args.norm_add, impl='default')
    ref_layer.cuda()
    ref_layer.half()
    ref_layer.reset_parameters()

    ref_inputs    = torch.randn(args.seq_length, args.num_seqs_start, args.hidden_dim, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
    ref_inputs_kv = None
    if args.encdec_attn :
        ref_inputs_kv    = torch.randn(args.seq_length, args.num_seqs_start, args.hidden_dim, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

    ref_grads         = torch.randn_like(ref_inputs)

    ref_outputs,_ = ref_layer.forward(ref_inputs,
                                      ref_inputs_kv,
                                      ref_inputs_kv,
                                      key_padding_mask=None,
                                      need_weights=False,
                                      attn_mask=None,
                                      is_training=(not args.eval))

    ref_outputs.backward(ref_grads)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tst_layer = None
    if args.encdec_attn :
        tst_layer = EncdecMultiheadAttn(args.hidden_dim, args.heads, dropout=dropout_prob, bias=False, include_norm_add=args.norm_add, impl='fast')
    else:
        tst_layer = SelfMultiheadAttn(args.hidden_dim, args.heads, dropout=dropout_prob, bias=False, include_norm_add=args.norm_add, impl='fast')
    tst_layer.cuda()
    tst_layer.half()
    tst_layer.reset_parameters()

    tst_inputs    = torch.randn(args.seq_length, args.num_seqs_start, args.hidden_dim, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
    tst_inputs_kv = None
    if args.encdec_attn :
        tst_inputs_kv    = torch.randn(args.seq_length, args.num_seqs_start, args.hidden_dim, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

    assert torch.equal(ref_inputs,tst_inputs), "ERROR: Inputs are different!"

    tst_grads         = torch.randn_like(tst_inputs)

    tst_outputs,_ = tst_layer.forward(tst_inputs,
                                      tst_inputs_kv,
                                      tst_inputs_kv,
                                      key_padding_mask=None,
                                      need_weights=False,
                                      attn_mask=None,
                                      is_training=(not args.eval))

    tst_outputs.backward(tst_grads)

    fwd_close = torch.equal(ref_outputs, tst_outputs)
    bwd_close = torch.equal(ref_inputs.grad, tst_inputs.grad)

    diff_fwd = ref_outputs - tst_outputs
    diff_cnt_fwd = diff_fwd.ne(0.0).sum()
    diff_accum_fwd = diff_fwd.abs().sum()

    diff_bwd = ref_inputs.grad - tst_inputs.grad
    diff_cnt_bwd = diff_bwd.ne(0.0).sum()
    diff_accum_bwd = diff_bwd.abs().sum()

    print(">>> Seed: ", seed, fwd_close, diff_cnt_fwd.item(), diff_accum_fwd.item(), bwd_close, diff_cnt_bwd.item(), diff_accum_bwd.item())
