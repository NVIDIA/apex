import inspect
import os
import torch
import torch.nn.functional as F
import unittest

from apex import pyprof
pyprof.nvtx.init()

# TODO: add tests for:
# F.bilinear, F.l1_loss, F.multilabel_soft_margin_loss, F.multi_margin_loss

class TestPyProfNvtx(unittest.TestCase):

    def __init__(self, testName, dtype=torch.float16):
        super().__init__(testName) 
        self.dtype = dtype

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_conv1d(self):
        # Data and weight tensors
        tensor1d_in_conv = torch.randn(32, 3, 224, device='cuda', dtype=self.dtype)
        tensor1d_in_conv_grouped = torch.randn(32, 6, 224, device='cuda', dtype=self.dtype)
        conv1d_filter = torch.randn(16, 3, 3, device='cuda', dtype=self.dtype)
        conv1d_bias = torch.ones(16, device='cuda', dtype=self.dtype)
        # Vanilla conv1d
        conv1d_out_vanilla = F.conv1d(tensor1d_in_conv, conv1d_filter)
        # conv1d with bias
        conv1d_out_with_bias = F.conv1d(tensor1d_in_conv, conv1d_filter, bias=conv1d_bias)
        # conv1d - stride > 1
        conv1d_out_strided = F.conv1d(tensor1d_in_conv, conv1d_filter, stride=2)
        # conv1d - dilation > 1
        conv1d_out_dilated = F.conv1d(tensor1d_in_conv, conv1d_filter, dilation=2)
        # conv1d - groups > 1
        conv1d_out_grouped = F.conv1d(tensor1d_in_conv_grouped, conv1d_filter, groups=2)
        # conv1d - padding with zeros
        conv1d_out_padding_zeros = F.conv1d(tensor1d_in_conv, conv1d_filter, padding=6)
    
    def test_conv2d(self):
        # Data and weight tensors
        tensor2d_in_conv = torch.randn(32, 3, 224, 224, device='cuda', dtype=self.dtype)
        tensor2d_in_conv_grouped = torch.randn(32, 6, 224, 224, device='cuda', dtype=self.dtype)
        conv2d_filter = torch.randn(16, 3, 3, 3, device='cuda', dtype=self.dtype)
        conv2d_bias = torch.ones(16, device='cuda', dtype=self.dtype)
        # Vanilla conv2d
        conv2d_out_vanilla = F.conv2d(tensor2d_in_conv, conv2d_filter)
        # conv2d with bias
        conv2d_with_bias = F.conv2d(tensor2d_in_conv, conv2d_filter, bias=conv2d_bias)
        # conv2d - stride > 1
        conv2d_out_strided = F.conv2d(tensor2d_in_conv, conv2d_filter, stride=2)
        # conv2d - dilation > 1
        conv2d_out_dilated = F.conv2d(tensor2d_in_conv, conv2d_filter, dilation=2)
        # conv2d - groups > 1
        conv2d_out_grouped = F.conv2d(tensor2d_in_conv_grouped, conv2d_filter, groups=2)
        # conv2d - padding with zeros
        conv2d_out_padding_zeros = F.conv2d(tensor2d_in_conv, conv2d_filter, padding=6)
    
    
    def test_conv3d(self):
        # Data and weight tensors
        tensor3d_in_conv = torch.randn(32, 3, 16, 224, 224, device='cuda', dtype=self.dtype)
        tensor3d_in_conv_grouped = torch.randn(32, 6, 16, 224, 224, device='cuda', dtype=self.dtype)
        conv3d_filter = torch.randn(16, 3, 3, 3, 3, device='cuda', dtype=self.dtype)
        conv3d_bias = torch.ones(16, device='cuda', dtype=self.dtype)
        # Vanilla conv3d
        conv3d_out_vanilla = F.conv3d(tensor3d_in_conv, conv3d_filter)
        # conv3d - stride > 1
        conv3d_out_strided = F.conv3d(tensor3d_in_conv, conv3d_filter, stride=2)
        # conv3d - dilation > 1
        conv3d_out_dilated = F.conv3d(tensor3d_in_conv, conv3d_filter, dilation=2)
        # conv3d - groups > 1
        conv3d_out_grouped = F.conv3d(tensor3d_in_conv_grouped, conv3d_filter, groups=2)
        # conv3d - padding with zeros
        conv3d_out_padding_zeros = F.conv3d(tensor3d_in_conv, conv3d_filter, padding=6)
    
    def test_conv_transpose1d(self):
        # Data and weight tensors
        conv_transpose1d_tensor = torch.randn(64, 16, 64, device='cuda', dtype=self.dtype)
        conv_transpose1d_filter = torch.randn(16, 32, 3, device='cuda', dtype=self.dtype)
        conv_transpose1d_bias = torch.randn(32, device='cuda', dtype=self.dtype)
        # Conv transpose runs
        conv_transpose1d_out = F.conv_transpose1d(conv_transpose1d_tensor, conv_transpose1d_filter)
        conv_transpose1d_out_biased = F.conv_transpose1d(conv_transpose1d_tensor, conv_transpose1d_filter, bias=conv_transpose1d_bias)
        conv_transpose1d_out_strided = F.conv_transpose1d(conv_transpose1d_tensor, conv_transpose1d_filter, stride=2)
        conv_transpose1d_out_padded = F.conv_transpose1d(conv_transpose1d_tensor, conv_transpose1d_filter, padding=3)
        conv_transpose1d_out2_padded = F.conv_transpose1d(conv_transpose1d_tensor, conv_transpose1d_filter, output_padding=2, dilation=3)
        conv_transpose1d_out_grouped = F.conv_transpose1d(conv_transpose1d_tensor, conv_transpose1d_filter, groups=2)
        conv_transpose1d_out_dilated = F.conv_transpose1d(conv_transpose1d_tensor, conv_transpose1d_filter, dilation=2)
    
    
    def test_conv_transpose2d(self):
        # Data and weight tensors
        conv_transpose2d_tensor = torch.randn(64, 8, 5, 5, device='cuda', dtype=self.dtype)
        conv_transpose2d_filter = torch.randn(8, 16, 3, 3, device='cuda', dtype=self.dtype)
        conv_transpose2d_bias = torch.randn(16, device='cuda', dtype=self.dtype)
        # Conv transpose runs
        conv_transpose2d_out = F.conv_transpose2d(conv_transpose2d_tensor, conv_transpose2d_filter)
        conv_transpose2d_out_biased = F.conv_transpose2d(conv_transpose2d_tensor, conv_transpose2d_filter, bias=conv_transpose2d_bias)
        conv_transpose2d_out_strided = F.conv_transpose2d(conv_transpose2d_tensor, conv_transpose2d_filter, stride=2)
        conv_transpose2d_out_padded = F.conv_transpose2d(conv_transpose2d_tensor, conv_transpose2d_filter, padding=3)
        conv_transpose2d_out2_padded = F.conv_transpose2d(conv_transpose2d_tensor, conv_transpose2d_filter, output_padding=2, dilation=3)
        conv_transpose2d_out_grouped = F.conv_transpose2d(conv_transpose2d_tensor, conv_transpose2d_filter, groups=2)
        conv_transpose2d_out_dilated = F.conv_transpose2d(conv_transpose2d_tensor, conv_transpose2d_filter, dilation=2)
    
    def test_conv_transpose3d(self):
        # Data and weight tensors
        conv_transpose3d_tensor = torch.randn(20, 16, 50, 10, 20, device='cuda', dtype=self.dtype)
        conv_transpose3d_filter = torch.randn(16, 33, 3, 3, 3, device='cuda', dtype=self.dtype)
        conv_transpose3d_bias = torch.randn(33, device='cuda', dtype=self.dtype)
        # Conv transpose runs
        conv_transpose3d_out = F.conv_transpose3d(conv_transpose3d_tensor, conv_transpose3d_filter)
        conv_transpose3d_out_biased = F.conv_transpose3d(conv_transpose3d_tensor, conv_transpose3d_filter, bias=conv_transpose3d_bias)
        conv_transpose3d_out_strided = F.conv_transpose3d(conv_transpose3d_tensor, conv_transpose3d_filter, stride=2)
        conv_transpose3d_out_padded = F.conv_transpose3d(conv_transpose3d_tensor, conv_transpose3d_filter, padding=3)
        conv_transpose3d_out2_padded = F.conv_transpose3d(conv_transpose3d_tensor, conv_transpose3d_filter, output_padding=2, dilation=3)
        conv_transpose3d_out_grouped = F.conv_transpose3d(conv_transpose3d_tensor, conv_transpose3d_filter, groups=2)
        conv_transpose3d_out_dilated = F.conv_transpose3d(conv_transpose3d_tensor, conv_transpose3d_filter, dilation=2)
    
    def test_unfold(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        kernel_size = (4, 5)
        inp_unf_dilated = F.unfold(inp, kernel_size, dilation=2)
        inp_unf_padded = F.unfold(inp, kernel_size, padding=2)
        inp_unf_strided = F.unfold(inp, kernel_size, stride=2)
    
    def test_fold(self):
        inp = torch.randn(3, 20, 20, device='cuda', dtype=self.dtype)
        inp_folded = F.fold(inp, (4, 5), (1, 1))
    
    def test_avg_pool1d(self):
        inp = torch.randn(1, 1, 28, device='cuda', dtype=self.dtype)
        out = F.avg_pool1d(inp, kernel_size=5, stride=2, padding=2, ceil_mode=True, count_include_pad=False)
    
    def test_avg_pool2d(self):
        inp = torch.randn(1, 3, 224, 224, device='cuda', dtype=self.dtype)
        out = F.avg_pool2d(inp, kernel_size=5, stride=2, padding=2, ceil_mode=True, count_include_pad=False)
    
    def test_avg_pool3d(self):
        inp = torch.randn(1, 3, 16, 224, 224, device='cuda', dtype=self.dtype)
        out = F.avg_pool3d(inp, kernel_size=5, stride=2, padding=2, ceil_mode=True, count_include_pad=False)
    
    def test_adaptive_avg_pool1d(self):
        inp = torch.randn(1, 1, 28, device='cuda', dtype=self.dtype)
        out = F.adaptive_avg_pool1d(inp, output_size=5) 
    
    def test_adaptive_avg_pool2d(self):
        inp = torch.randn(1, 16, 32, 32, device='cuda', dtype=self.dtype)
        out = F.adaptive_avg_pool2d(inp, output_size=5) 
    
    def test_adaptive_avg_pool3d(self):
        inp = torch.randn(1, 16, 16, 32, 32, device='cuda', dtype=self.dtype)
        out = F.adaptive_avg_pool3d(inp, output_size=5) 
    
    def test_max_pool1d(self):
        inp = torch.randn(1, 16, 32, device='cuda', dtype=self.dtype)
        out = F.max_pool1d(inp, kernel_size=5, stride=2, padding=2, return_indices=True, ceil_mode=True)
    
    def test_max_pool2d(self):
        inp = torch.randn(1, 16, 32, 32, device='cuda', dtype=self.dtype)
        out = F.max_pool2d(inp, kernel_size=5, stride=2, padding=2, return_indices=True, ceil_mode=True)
    
    def test_max_pool3d(self):
        inp = torch.randn(1, 16, 16, 32, 32, device='cuda', dtype=self.dtype)
        out = F.max_pool3d(inp, kernel_size=5, stride=2, padding=2, return_indices=True, ceil_mode=True)
    
    def test_adaptive_max_pool1d(self):
        inp = torch.randn(1, 16, 28, device='cuda', dtype=self.dtype)
        out = F.adaptive_max_pool1d(inp, output_size=5, return_indices=True) 
    
    def test_adaptive_max_pool2d(self):
        inp = torch.randn(1, 16, 32, 32, device='cuda', dtype=self.dtype)
        out = F.adaptive_max_pool2d(inp, output_size=5, return_indices=True) 
    
    def test_adaptive_max_pool3d(self):
        inp = torch.randn(1, 16, 16, 32, 32, device='cuda', dtype=self.dtype)
        out = F.adaptive_max_pool3d(inp, output_size=5, return_indices=True) 
    
    def test_max_unpool1d(self):
        inp = torch.randn(1, 16, 32, device='cuda', dtype=self.dtype)
        output, indices = F.max_pool1d(inp, kernel_size=5, stride=2, padding=2, return_indices=True, ceil_mode=True)
        output = F.max_unpool1d(output, indices, kernel_size=2, stride=2, padding=2)
    
    def test_max_unpool2d(self):
        inp = torch.randn(1, 16, 32, 32, device='cuda', dtype=self.dtype)
        output, indices = F.max_pool2d(inp, kernel_size=5, stride=2, padding=2, return_indices=True, ceil_mode=True)
        output = F.max_unpool2d(output, indices, kernel_size=2, stride=2, padding=2)
    
    def test_max_unpool3d(self):
        inp = torch.randn(1, 16, 8, 32, 32, device='cuda', dtype=self.dtype)
        output, indices = F.max_pool3d(inp, kernel_size=5, stride=2, padding=2, return_indices=True, ceil_mode=True)
        output = F.max_unpool3d(output, indices, kernel_size=2, stride=2, padding=2)
    
    def test_lp_pool1d(self):
        inp = torch.randn(1, 32, 64, device='cuda', dtype=self.dtype)
        output = F.lp_pool1d(inp, 2, 3, stride=2, ceil_mode=True)
    
    def test_lp_pool2d(self):
        #torch.nn.LPPool2d(norm_type, kernel_size, stride=None, ceil_mode=False)
        inp = torch.randn(1, 32, 64, 64, device='cuda', dtype=self.dtype)
        output = F.lp_pool2d(inp, 2, 3, stride=2, ceil_mode=True)
    
    def test_threshold(self):
        inp = torch.randn(1, 8, 32, 32, device='cuda', dtype=self.dtype)
        output = F.threshold(inp, 6, 6, inplace=False)
    
    def test_threshold_(self):
        inp = torch.randn(1, 8, 32, 32, device='cuda', dtype=self.dtype)
        output = F.threshold_(inp, 6, 6)
    
    def test_relu(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.relu(inp, inplace=False)
    
    def test_relu_(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.relu_(inp)
    
    def test_hardtanh(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.hardtanh(inp, min_val=-1., max_val=1., inplace=False)
    
    def test_hardtanh_(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.hardtanh_(inp, min_val=-1., max_val=1.)
    
    def test_relu6(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.relu6(inp, inplace=False)
    
    def test_elu(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.elu(inp, alpha=1.0, inplace=False)
    
    def test_elu_(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.elu_(inp, alpha=1.0)
    
    def test_selu(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.selu(inp)
    
    def test_celu(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.celu(inp, alpha=1.0, inplace=False)
    
    def test_leaky_relu(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.leaky_relu(inp, negative_slope=0.01, inplace=False)
    
    def test_leaky_relu_(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.leaky_relu_(inp, negative_slope=0.01)
    
    def test_prelu(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        weight = torch.randn(1, device='cuda', dtype=self.dtype)
        output = F.prelu(inp, weight)
    
    def test_rrelu(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.rrelu(inp, lower=1./8, upper=1./3, training=False, inplace=False)
    
    def test_rrelu_(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.rrelu(inp, lower=1./8, upper=1./3, training=False)
    
    def test_glu(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.glu(inp, dim=-1)
    
    def test_logsigmoid(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.logsigmoid(inp)
    
    def test_hardshrink(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.hardshrink(inp, lambd=0.5)
    
    def test_tanhshrink(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.tanhshrink(inp)
    
    def test_softsign(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.softsign(inp)
    
    def test_softplus(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.softplus(inp, beta=1, threshold=20)
    
    def test_softmin(self):
        inp = torch.randn(16, 1024, device='cuda', dtype=self.dtype)
        output = F.softmin(inp, dim=1,  _stacklevel=3, dtype=self.dtype)
    
    def test_softmax(self):
        inp = torch.randn(16, 1024, device='cuda', dtype=self.dtype)
        output = F.softmax(inp, dim=1, _stacklevel=3, dtype=self.dtype)
    
    def test_softshrink(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.softshrink(inp, lambd=0.5)
    
    def test_gumbel_softmax(self):
        inp = torch.randn(16, 1024, device='cuda', dtype=self.dtype)
        output = F.gumbel_softmax(inp, tau=1, hard=False, eps=1e-10, dim=-1)
    
    def test_log_softmax(self):
        inp = torch.randn(16, 1024, device='cuda', dtype=self.dtype)
        output = F.log_softmax(inp, dim=-1, _stacklevel=3)
    
    def test_tanh(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = torch.tanh(inp)
    
    def test_sigmoid(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = torch.sigmoid(inp)
    
    def test_batch_norm(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        # running_mean, running_var
        running_mean = torch.randn(3, device='cuda', dtype=self.dtype)
        running_var = torch.randn(3, device='cuda', dtype=self.dtype)
        output = F.batch_norm(inp, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)
    
    def test_instance_norm(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        running_mean = torch.randn(3, device='cuda', dtype=self.dtype)
        running_var = torch.randn(3, device='cuda', dtype=self.dtype)
        output = F.instance_norm(inp, running_mean=running_mean, running_var=running_var, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05)
    
    def test_layer_norm(self):
        inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
        output = F.layer_norm(inp, inp.size()[1:], weight=None, bias=None, eps=1e-05)
    
    def test_local_response_norm(self):
        inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
        output = F.local_response_norm(inp, 2, alpha=0.0001, beta=0.75, k=1.0)
    
    def test_normalize(self):
        inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
        output = F.normalize(inp, p=2, dim=1, eps=1e-12, out=None)
    
    def test_linear(self):
        inp = torch.randn(32, 64, 128, device='cuda', dtype=self.dtype)
        weight = torch.randn(256, 128, device='cuda', dtype=self.dtype)
        output = F.linear(inp, weight, bias=None)
    
    def test_dropout(self):
        inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
        output = F.dropout(inp, p=0.5, training=True, inplace=False)
    
    def test_alpha_dropout(self):
        inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
        output = F.alpha_dropout(inp, p=0.5, training=True, inplace=False)
    
    def test_dropout2d(self):
        inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
        output = F.dropout2d(inp, p=0.5, training=True, inplace=False)
    
    def test_dropout3d(self):
        inp = torch.randn(16, 8, 32, 64, 64, device='cuda', dtype=self.dtype)
        output = F.dropout3d(inp, p=0.5, training=True, inplace=False)
    
    def test_embedding(self):
        pre_embed_dim = 1024
        post_embed_dim = 32
        inp = torch.randint(0, pre_embed_dim, (128, 16), device='cuda')    
        weight = torch.randn(pre_embed_dim, post_embed_dim, device='cuda', dtype=self.dtype)
        output = F.embedding(inp, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
    
    def test_embedding_bag(self):
        pre_embed_dim = 1024
        post_embed_dim = 32
        inp = torch.randint(0, pre_embed_dim, (128, 16), device='cuda')    
        weight = torch.randn(pre_embed_dim, post_embed_dim, device='cuda', dtype=self.dtype)
        output = F.embedding_bag(inp, weight, offsets=None, max_norm=None, norm_type=2,
            scale_grad_by_freq=False, mode='mean', sparse=False)
    
    def test_one_hot(self):
        num_classes = 10
        inp = torch.randint(0, num_classes, (128, 16), device='cuda')    
        output = F.one_hot(inp, num_classes=10) 
    
    def test_pairwise_distance(self):
        inp1 = torch.randn(1024, 128, device='cuda', dtype=self.dtype)
        inp2 = torch.randn(1024, 128, device='cuda', dtype=self.dtype)
        output = F.pairwise_distance(inp1, inp2, p=2.0, eps=1e-06, keepdim=False) 
    
    def test_cosine_similarity(self):
        inp1 = torch.randn(1024, 128, device='cuda', dtype=self.dtype)
        inp2 = torch.randn(1024, 128, device='cuda', dtype=self.dtype)
        output = F.cosine_similarity(inp1, inp2, dim=1, eps=1e-8)
    
    def test_pdist(self):
        # pdist is not implemented for fp16
        inp = torch.randn(128, 128, device='cuda', dtype=torch.float32)
        output = F.pdist(inp, p=2)
    
    def test_binary_cross_entropy(self):
        # binary_cross_entropy is not implemented for fp16
        inp = torch.randn(32, 128, device='cuda', dtype=torch.float32, requires_grad=True)
        target = torch.randn(32, 128, device='cuda', dtype=torch.float32, requires_grad=False)
        output = F.binary_cross_entropy(torch.sigmoid(inp), target)
    
    def test_binary_cross_entropy_with_logits(self):
        inp = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        target = torch.empty_like(inp).random_(2)
        output = F.binary_cross_entropy_with_logits(inp, target)
    
    def test_poisson_nll_loss(self):
        inp = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        target = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=False)
        output = F.poisson_nll_loss(inp, target, log_input=True, full=False,
            size_average=None, eps=1e-08, reduce=None, reduction='mean')
    
    def test_cosine_embedding_loss(self):
        inp1 = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        inp2 = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        target = torch.randn(32, device='cuda', dtype=self.dtype, requires_grad=False)
        output = F.cosine_embedding_loss(inp1, inp2, target, margin=0,
            size_average=None, reduce=None, reduction='mean')
    
    def test_cross_entropy(self):
        inp = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        target = torch.randint(0, 100, (32,), device='cuda', dtype=torch.long, requires_grad=False)
        output = F.cross_entropy(inp, target, weight=None, size_average=None,
            ignore_index=-100, reduce=None, reduction='mean')
    
    def test_ctc_loss(self):
        # force fp32 because _th_normal_ (used by next line is not supported for fp16)
        log_probs = torch.randn(50, 16, 20, device='cuda', dtype=torch.float32).log_softmax(2).detach().requires_grad_()
        targets = torch.randint(1, 20, (16, 30), device='cuda', dtype=torch.long)
        input_lengths = torch.full((16,), 50, dtype=torch.long)
        target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    
    def test_hinge_embedding_loss(self):
        inp = torch.randn(128, 32, device='cuda', dtype=self.dtype)
        target = torch.randint(0, 1, (32,), device='cuda') - 1
        output = F.hinge_embedding_loss(inp, target, margin=1.0, size_average=None, reduce=None, reduction='mean') 
    
    def test_kl_div(self):
        inp = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        target = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        output = F.kl_div(inp, target, size_average=None, reduce=None, reduction='batchmean')
    
    def test_mse_loss(self):
        inp = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        target = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        output = F.mse_loss(inp, target, size_average=None, reduce=None, reduction='mean')
    
    def test_margin_ranking_loss(self):
        inp1 = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        inp2 = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        target = (torch.randint(0, 1, (128,), device='cuda') - 1).type_as(inp1)
        output = F.margin_ranking_loss(inp1, inp2, target, margin=0, size_average=None, reduce=None, reduction='mean')
    
    def test_multilabel_margin_loss(self):
        inp = torch.randn(1024, device='cuda', dtype=self.dtype, requires_grad=True)
        target = torch.randint(0, 10, (1024,), dtype=torch.long, device='cuda')
        output = F.multilabel_margin_loss(inp, target, size_average=None, reduce=None, reduction='mean')
    
    def test_nll_loss(self):
        inp = torch.randn(64, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        target = torch.randint(0, 10, (64,), device='cuda', dtype=torch.long) 
        output = F.nll_loss(inp, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    
    def test_smooth_l1_loss(self):
        inp = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        target = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=False)
        output = F.smooth_l1_loss(inp, target, size_average=None, reduce=None, reduction='mean')
    
    def test_soft_margin_loss(self):
        inp = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        target = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=False)
        output = F.soft_margin_loss(inp, target, size_average=None, reduce=None, reduction='mean') 
    
    def test_triplet_margin_loss(self):
        inp1 = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        inp2 = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        inp3 = torch.randn(32, 128, device='cuda', dtype=self.dtype, requires_grad=True)
        output = F.triplet_margin_loss(inp1, inp2, inp3, margin=1.0, p=2,
             eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
    
    def test_pixel_shuffle(self):
        inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
        output = torch.nn.functional.pixel_shuffle(inp, 2)
    
    def test_pad(self):
        inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
        pad = (3, 3)
        output = F.pad(inp, pad, mode='constant', value=0)
    
    def test_interpolate(self):
        inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
        output = F.interpolate(inp, size=None, scale_factor=2, mode='nearest', align_corners=None)
    
    def test_grid_sample(self):
        inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
        grid = torch.randn(16, 32, 32, 2, device='cuda', dtype=self.dtype)
        output = F.grid_sample(inp, grid, mode='bilinear', padding_mode='zeros')
    
    def test_affine_grid(self):
        theta = torch.randn(32, 2, 3, device='cuda', dtype=self.dtype)
        size = (32, 8, 32, 32)
        output = F.affine_grid(theta, size)


def run_tests(precision):
    dummy = TestPyProfNvtx('test_affine_grid', None)
    test_cases = list(filter(lambda x: 'test_' in x, map(lambda x: x[0], inspect.getmembers(dummy, predicate=inspect.ismethod))))
    print("Running tests for {}".format(precision))
    suite = unittest.TestSuite()
    for test_case in test_cases:
        suite.addTest(TestPyProfNvtx(test_case, precision))
    unittest.TextTestRunner().run(suite)

if __name__ == '__main__':
    run_tests(torch.float32)
    run_tests(torch.float16)
