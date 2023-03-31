import torch
import torch.onnx
from apex.contrib.sparsity.permutation_lib import Permutation

"""
Functional and behavioral correctness checking for network permutations
Each test class is a torch.nn.Module with three required members:
- self.input_shape is used to populate a dummy input
- self.expected_C_params indicates how many parameters are expected to be permuted in the C dimension
- self.expected_K_params indicates how many parameters are expected to be permuted in the K dimension

A test is successful if and only if:
1. The output of the un-permuted module matches (within a tolerance) the ouput of the permuted module
2. The number of parameters permuted in C, as reported by the Permutation class, matches the expected value in the test module
3. The number of parameters permuted in K, as reported by the Permutation class, matches the expected value in the test module

This file has all the test modules defined first, followed by the common test routine to check each module's correctness, and finally the main/entry point.
"""

class simple_convs(torch.nn.Module):
    """Stack of 2d convolutions with different normalization and activation functions"""

    def __init__(
        self,
        num_convs: int,
        channels: int,
        normalization: str = 'none',
        activation: str = 'ReLU',
    ):
        super().__init__()
        self.num_convs = num_convs
        self.channels = channels
        self.normalization = normalization
        self.activation = activation

        self.input_shape = [4, channels, 7, 7]

        # we'll permute all convs' weights along C except the first
        self.expected_C_params = -1
        self.expected_K_params = 0

        self.conv_stack = torch.nn.Sequential()
        for c in range(self.num_convs-1):
            self.conv_stack.add_module(f"conv_{c}", torch.nn.Conv2d(self.channels, self.channels, kernel_size=(3,3), padding=1))
            self.expected_C_params += 1
            self.expected_K_params += 2

            if self.normalization == 'BatchNorm2d':
                self.conv_stack.add_module(f"norm_{c}", torch.nn.BatchNorm2d(self.channels, track_running_stats=False))
                self.expected_K_params += 2
            elif self.normalization == 'LazyBatchNorm2d':
                self.conv_stack.add_module(f"norm_{c}", torch.nn.LazyBatchNorm2d(track_running_stats=False))
                self.expected_K_params += 2
            elif self.normalization == 'GroupNorm':
                self.conv_stack.add_module(f"norm_{c}", torch.nn.GroupNorm(4, self.channels, affine=True))
                self.expected_C_params -= 1 # GN prevents permutations of the neighboring convs
                self.expected_K_params -= 2
            elif self.normalization == 'InstanceNorm2d':
                self.conv_stack.add_module(f"norm_{c}", torch.nn.InstanceNorm2d(self.channels, affine=True, track_running_stats=False))
                self.expected_K_params += 2
            elif self.normalization == 'LocalResponseNorm':
                self.conv_stack.add_module(f"norm_{c}", torch.nn.LocalResponseNorm(16))
            elif self.normalization == 'LayerNorm1':
                self.conv_stack.add_module(f"norm_{c}", torch.nn.LayerNorm(7))
            elif self.normalization == 'LayerNorm2':
                self.conv_stack.add_module(f"norm_{c}", torch.nn.LayerNorm([7, 7]))
            elif self.normalization == 'LayerNorm3':
                self.conv_stack.add_module(f"norm_{c}", torch.nn.LayerNorm([self.channels, 7, 7]))
                self.expected_K_params += 2
            elif self.normalization == 'SyncBatchNorm':
                self.conv_stack.add_module(f"norm_{c}", torch.nn.SyncBatchNorm(self.channels, track_running_stats=False))
                self.expected_K_params += 2

            self.conv_stack.add_module(f"act_{c}", torch.nn.ReLU())

        self.conv_stack.add_module("conv_out", torch.nn.Conv2d(self.channels, 8, kernel_size=(1,1)))
        self.expected_C_params += 1

    def forward(self, x: torch.Tensor):

        x = self.conv_stack(x)

        return x

class conv_1d(torch.nn.Module):
    """1D convolutions in isolation and with siblings"""

    def __init__(
        self,
        with_2d = False,
    ):
        super().__init__()
        self.input_shape = [4, 16, 7, 7]
        self.expected_C_params = 0
        self.expected_K_params = 0
        self.with_2d = with_2d

        self.input_conv = torch.nn.Conv2d(self.input_shape[1], 32, kernel_size=(3,3), padding=1)
        self.expected_K_params += 2

        self.branch_a_1D = torch.nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.expected_C_params += 1
        self.expected_K_params += 2
        if self.with_2d:
            self.branch_b_2D = torch.nn.Conv2d(32, 32, kernel_size=(3,3), padding=1)
            self.expected_C_params += 1
            self.expected_K_params += 2

        self.out_conv = torch.nn.Conv2d(32, 8, kernel_size=(1,1))
        self.expected_C_params += 1

    def forward(self, x: torch.Tensor):
        
        step0 = self.input_conv(x)

        s0shape = step0.shape
        step1 = self.branch_a_1D(step0.view(s0shape[0], s0shape[1], s0shape[2]*s0shape[3])).view(s0shape)
        if self.with_2d:
            step1 = step1 + self.branch_b_2D(step0)

        return self.out_conv(step1)

class grouped_convs(torch.nn.Module):
    """Stack of 2d convolutions with different types of grouped convolutions"""

    def __init__(
        self,
    ):
        super().__init__()
        self.channels = 128
        self.input_shape = [4, self.channels, 7, 7]
        self.expected_C_params = 0
        self.expected_K_params = 0

        self.conv_stack = torch.nn.Sequential()
        self.conv_stack.add_module("conv_in", torch.nn.Conv2d(self.channels, self.channels, kernel_size=(3,3), padding=1))

        # dw conv will let previous and this layers' weights and biases permute along K
        self.expected_K_params += 4
        self.conv_stack.add_module("conv_dw", torch.nn.Conv2d(self.channels, self.channels, kernel_size=(3,3), padding=1, groups=self.channels))

        # regular conv permutes both
        self.expected_C_params += 1
        self.expected_K_params += 2
        self.conv_stack.add_module("conv_0", torch.nn.Conv2d(self.channels, self.channels, kernel_size=(3,3), padding=1, groups=1))    # explicit '1' groups for extra coverage

        # only 2 groups should allow permutations only in C
        self.expected_C_params += 1
        self.conv_stack.add_module("conv_gr2", torch.nn.Conv2d(self.channels, self.channels, kernel_size=(3,3), padding=1, groups=2))

        # another regular conv, this one can't do anything
        self.conv_stack.add_module("conv_1", torch.nn.Conv2d(self.channels, self.channels, kernel_size=(3,3), padding=1))

        # finally, grouped conv with small groups
        self.conv_stack.add_module("conv_gr64", torch.nn.Conv2d(self.channels, self.channels, kernel_size=(3,3), padding=1, groups=self.channels//2))

    def forward(self, input: torch.Tensor):
        
        return self.conv_stack(input)

class simple_forks_joins(torch.nn.Module):
    """Some simple residual connections to test collecting parameters into a single group.  Four sections: input, blocka + residual, blockb + blockc, output"""

    def __init__(
        self,
    ):
        super().__init__()
        self.channels = 64
        self.input_shape = [4, self.channels, 7, 7]
        self.expected_C_params = 0
        self.expected_K_params = 0

        self.input_convs = torch.nn.Sequential()
        # input conv can only permute along K
        self.expected_K_params += 2
        self.input_convs.add_module("conv_in0", torch.nn.Conv2d(self.channels, self.channels, kernel_size=(3,3), padding=1))
        # the next conv can permute along both C and K
        self.expected_C_params += 1
        self.expected_K_params += 2
        self.input_convs.add_module("conv_in1", torch.nn.Conv2d(self.channels, self.channels, kernel_size=(3,3), padding=1))
        # BN will permute 2 more along K
        self.expected_K_params += 2
        self.input_convs.add_module("bn_in1", torch.nn.BatchNorm2d(self.channels, track_running_stats=False))

        self.block_a = torch.nn.Sequential()
        # cut channels in half, then back to full, two fully permutable convs
        self.expected_C_params += 2
        self.expected_K_params += 4
        self.block_a.add_module("conv_a0", torch.nn.Conv2d(self.channels, self.channels // 2, kernel_size=(3,3), padding=1))
        self.block_a.add_module("conv_a1", torch.nn.Conv2d(self.channels // 2, self.channels, kernel_size=(3,3), padding=1))

        self.block_b = torch.nn.Sequential()
        # cut channels in half, then back to full, two fully permutable convs
        self.expected_C_params += 2
        self.expected_K_params += 4
        self.block_b.add_module("conv_b0", torch.nn.Conv2d(self.channels, self.channels // 2, kernel_size=(3,3), padding=1))
        self.block_b.add_module("conv_b1", torch.nn.Conv2d(self.channels // 2, self.channels, kernel_size=(3,3), padding=1))

        self.block_c = torch.nn.Sequential()
        # cut channels in half, then back to full, two fully permutable convs
        self.expected_C_params += 2
        self.expected_K_params += 4
        self.block_c.add_module("conv_c0", torch.nn.Conv2d(self.channels, self.channels // 2, kernel_size=(3,3), padding=1))
        self.block_c.add_module("conv_c1", torch.nn.Conv2d(self.channels // 2, self.channels, kernel_size=(3,3), padding=1))

        self.output_conv = torch.nn.Sequential()
        self.expected_C_params += 1
        self.output_conv.add_module("conv_out", torch.nn.Conv2d(self.channels, 8, kernel_size=(3,3), padding=1))

    def forward(self, input: torch.Tensor):
        step0 = self.input_convs(input)
        step1 = step0 + self.block_a(step0)
        step2 = self.block_b(step1) + self.block_c(step1)
        return self.output_conv(step2)

class different_grouped_convs(torch.nn.Module):
    """Convolutions with different group sizes need to use the GCD of the input channel counts if siblings"""

    def __init__(
        self,
    ):
        super().__init__()
        self.channels = 16
        self.input_shape = [4, self.channels, 7, 7]
        self.expected_C_params = 0
        self.expected_K_params = 0

        self.input_conv = torch.nn.Sequential()
        self.expected_K_params += 2
        self.input_conv.add_module("input_conv", torch.nn.Conv2d(self.channels, 128, kernel_size=(3,3), padding=1))

        self.expected_C_params += 4
        # 4 parallel blocks with decreasing group size from "left" to "right"
        self.block_a = torch.nn.Sequential()
        self.block_a.add_module("conv_a", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1))
        self.block_b = torch.nn.Sequential()
        self.block_b.add_module("conv_b", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, groups=2))
        self.block_c = torch.nn.Sequential()
        self.block_c.add_module("conv_c", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, groups=4))
        self.block_d = torch.nn.Sequential()
        self.block_d.add_module("conv_d", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, groups=8))

        # output can't permute along C, disallowed by parents
        self.output_conv = torch.nn.Sequential()
        self.output_conv.add_module("output_conv", torch.nn.Conv2d(128, 8, kernel_size=(3,3), padding=1))

    def forward(self, input: torch.Tensor):
        step0 = self.input_conv(input)
        step1 = self.block_a(step0) + self.block_b(step0) + self.block_c(step0) + self.block_d(step0)
        return self.output_conv(step1)

class siblings_poison(torch.nn.Module):
    """A single sibling that cannot permute along C poisons all other siblings in its group"""

    def __init__(
        self,
    ):
        super().__init__()
        self.input_shape = [4, 16, 7, 7]
        self.expected_C_params = 0
        self.expected_K_params = 0

        self.input_conv = torch.nn.Sequential()
        self.input_conv.add_module("input_conv", torch.nn.Conv2d(self.input_shape[1], 128, kernel_size=(3,3), padding=1))

        # two parallel block: conv->flatten->linear | flatten->linear
        self.expected_K_params += 4 # two linears will have their output channels permuted for the output layer
        self.block_a = torch.nn.Sequential()
        self.block_a.add_module("conv_a", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1))
        self.block_a.add_module("flatten_a", torch.nn.Flatten(1))
        self.block_a.add_module("linear_a", torch.nn.Linear(6272, 128))

        self.block_b = torch.nn.Sequential()
        self.block_b.add_module("flatten_b", torch.nn.Flatten(1))
        self.block_b.add_module("linear_b",  torch.nn.Linear(6272, 128))

        self.output = torch.nn.Sequential()
        self.expected_C_params += 1 # output layer will have its C dimension permuted
        self.output.add_module("output", torch.nn.Linear(128, 8))

    def forward(self, input: torch.Tensor):
        step0 = self.input_conv(input)
        step1 = self.block_a(step0) + self.block_b(step0)
        return self.output(step1)

class coparent_poison(torch.nn.Module):
    """A single coparent that cannot permute along K poisons all other coparents in its group"""

    def __init__(
        self,
    ):
        super().__init__()
        self.input_shape = [4, 16, 7, 7]
        self.expected_C_params = 0
        self.expected_K_params = 0

        self.input_conv = torch.nn.Sequential()
        self.expected_K_params += 2
        self.input_conv.add_module("input_conv", torch.nn.Conv2d(self.input_shape[1], 128, kernel_size=(3,3), padding=1))

        # two parallel block: conv | conv-> grouped conv
        self.expected_C_params += 3  # all convs permute along C
        self.expected_K_params += 2  # only conv_b0 permutes along K
        self.block_a = torch.nn.Sequential()
        self.block_a.add_module("conv_a", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1))

        self.block_b = torch.nn.Sequential()
        self.block_b.add_module("conv_b0", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1))
        self.block_b.add_module("conv_b1", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, groups=4))

        self.output = torch.nn.Sequential()
        self.output.add_module("output", torch.nn.Conv2d(128, 8, kernel_size=(1,1)))

    def forward(self, input: torch.Tensor):
        step0 = self.input_conv(input)
        step1 = self.block_a(step0) + self.block_b(step0)
        return self.output(step1)
    

class depthwise_child_is_sibling(torch.nn.Module):
    """The child of a depthwise convolution should act as a sibling"""

    def __init__(
        self,
    ):
        super().__init__()
        self.input_shape = [4, 16, 7, 7]
        self.expected_C_params = 0
        self.expected_K_params = 0

        self.input_conv = torch.nn.Sequential()
        self.expected_K_params += 2
        self.input_conv.add_module("input_conv", torch.nn.Conv2d(self.input_shape[1], 128, kernel_size=(3,3), padding=1))

        # two parallel block: conv | depthwise->conv
        self.expected_C_params += 2
        self.expected_K_params += 4 + 2
        self.block_a = torch.nn.Sequential()
        self.block_a.add_module("conv_a", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1))

        self.block_b = torch.nn.Sequential()
        self.block_b.add_module("conv_b_dw", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, groups=128))
        self.block_b.add_module("conv_b_1",  torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1))

        self.output_conv = torch.nn.Sequential()
        self.expected_C_params += 1
        self.output_conv.add_module("output_conv", torch.nn.Conv2d(128, 8, kernel_size=(1,1)))

    def forward(self, input: torch.Tensor):
        step0 = self.input_conv(input)
        step1 = self.block_a(step0) + self.block_b(step0)
        return self.output_conv(step1)


class module_attribute(torch.nn.Module):
    """Attributes of some module must be permuted if they feed some operation that is permuted"""

    def __init__(
        self,
        complexity: int = 0,
    ):
        super().__init__()
        self.input_shape = [4, 16, 7, 7]
        self.expected_C_params = 0
        self.expected_K_params = 0
        self.complexity = complexity

        self.input_conv = torch.nn.Sequential()
        self.expected_K_params += 3 # conv weight, conv bias, input_offset C (counts as K since it's acting as a parent)
        self.input_offset = torch.nn.Parameter(torch.zeros(128,7,7))
        torch.nn.init.normal_(self.input_offset.data, mean=0.0, std=2.0)
        self.input_conv.add_module("conv_input", torch.nn.Conv2d(self.input_shape[1], 128, kernel_size=(3,3), padding=1))

        # add a couple more layers, and let the same offset affect another layer, as well
        if complexity == 1:
            self.expected_C_params += 2
            self.expected_K_params += 4
            self.stack_a = torch.nn.Sequential()
            self.stack_a.add_module("conv_a", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1))

            self.stack_b = torch.nn.Sequential()
            self.stack_b.add_module("conv_b", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1))


        self.output_conv = torch.nn.Sequential()
        self.expected_C_params += 1
        self.output_conv.add_module("conv_output", torch.nn.Conv2d(128, 8, kernel_size=(3,3)))

    def forward(self, input: torch.Tensor):
        batch_input_offset = self.input_offset.expand(input.shape[0], -1, -1, -1)
        x = self.input_conv(input) + batch_input_offset
        if self.complexity == 1:
            x = self.stack_a(x) + batch_input_offset
            x = self.stack_b(x) + batch_input_offset
        return self.output_conv(x)


class square_attribute(torch.nn.Module):
    """Attributes with multiple dimensions matching the permutation length should only be permuted along the correct dimension"""
    # TODO: currently, such an attribute will disallow permutations around it, but with effort, it could be handled correctly.

    def __init__(
        self,
    ):
        super().__init__()
        self.input_shape = [4, 16, 16]
        self.expected_C_params = 0
        self.expected_K_params = 0
        
        self.input_linear = torch.nn.Sequential()
        #self.expected_K_params += 2  # if handled correctly, the linear's K and the offset's K should both be permuted
        self.input_linear.add_module("linear_input", torch.nn.Linear(self.input_shape[1], 16))
        self.input_offset = torch.nn.Parameter(torch.zeros(16, 16))
        torch.nn.init.normal_(self.input_offset.data, mean=0.0, std=2.0)

        self.output_linear = torch.nn.Sequential()
        #self.expected_C_params += 1  # if handled correctly, this should be permuted
        self.output_linear.add_module("linear_output", torch.nn.Linear(16, 8))

    def forward(self, input: torch.Tensor):
        batch_input_offset = self.input_offset.expand(input.shape[0], -1, -1)
        x = self.input_linear(input) + torch.permute(batch_input_offset, (0, 2, 1))
        return self.output_linear(x)


class MHA_test(torch.nn.Module):
    """MultiheadAttention modules are unique, we need to check permutations for input and ouput projections"""

    def __init__(
        self,
        hidden_dim: int = 256,
        seq_len: int = 64,
        num_heads: int = 16
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.input_shape = [4, self.seq_len, self.hidden_dim]

        self.expected_C_params = 1
        self.expected_K_params = 2

        self.MHA0 = torch.nn.MultiheadAttention(self.hidden_dim, self.num_heads, dropout=False, batch_first=True)
        self.MHA1 = torch.nn.MultiheadAttention(self.hidden_dim, self.num_heads, dropout=False, batch_first=True)

    def forward(self, input: torch.Tensor):
        step0,_ = self.MHA0(input, input, input)
        step1,_ = self.MHA1(step0, step0, step0)
        return step1
        

class one_sparse_sibling(torch.nn.Module):
    """If only one of two siblings is sparse, both need to be permuted"""

    def __init__(
        self,
    ):
        super().__init__()
        self.input_shape = [4, 16, 7, 7]
        self.expected_C_params = 0
        self.expected_K_params = 0

        self.in_conv = torch.nn.Sequential()
        self.expected_K_params += 2
        self.in_conv.add_module("conv_in", torch.nn.Conv2d(self.input_shape[1], 128, kernel_size=(3,3), padding=1))

        self.block_a = torch.nn.Sequential()
        self.expected_C_params += 1 # only conv_a0 will be permuted along C
        self.expected_K_params += 2 # only conv_a1 will be permuted along K
        self.block_a.add_module("conv_a0", torch.nn.Conv2d(128, 3, kernel_size=(1,1)))
        self.block_a.add_module("conv_a1", torch.nn.Conv2d(3, 128, kernel_size=(3,3), padding=1))

        self.block_b = torch.nn.Sequential()
        self.expected_C_params += 2 # even though conv_a0 will not be sparse (only 3 output channels), conv_b0 can still be permuted along C
        self.expected_K_params += 4
        self.block_b.add_module("conv_b0", torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1))
        self.block_b.add_module("conv_b1", torch.nn.Conv2d(128, 128, kernel_size=(1,1)))

        self.out_conv = torch.nn.Sequential()
        self.expected_C_params += 1
        self.out_conv.add_module("conv_out", torch.nn.Conv2d(128, 8, kernel_size=(1,1)))

    def forward(self, input: torch.Tensor):
        step0 = self.in_conv(input)
        step1 = self.block_a(step0) + self.block_b(step0)
        return self.out_conv(step1)


class test_concat(torch.nn.Module):
    """If concats are along the channel dimension (dim1 of NCHW), downstream layers can still be permuted despite C!=parentK"""

    def __init__(
        self,
        ratio = 1,  # ratio between # channels in either path to be concatenated
        dim = 1,    # dimension to concatenate, K by default
        depth = 1,  # number of concats to stack
    ):
        super().__init__()
        assert dim == 1 or ratio == 1 ,"can't concat along dimensions other than K if K's don't match"
        self.dim = dim
        self.depth = depth
        self.input_shape = [4, 16, 7, 7]
        self.expected_C_params = 0
        self.expected_K_params = 0

        self.in_conv = torch.nn.Sequential()
        self.expected_K_params += 2
        self.in_conv.add_module("conv_in", torch.nn.Conv2d(self.input_shape[1], 64, kernel_size=(1,1)))

        self.left_paths = torch.nn.ModuleList([torch.nn.Conv2d(64, 64, kernel_size=(1,1))])
        self.expected_C_params += 1
        self.expected_K_params += 2

        in_C = 64
        out_C = 64
        for d in range(1,depth,1):
            self.expected_C_params += 1
            self.expected_K_params += 2
            if dim == 1:
                out_C += 64
            self.left_paths.append(torch.nn.Conv2d(in_C+64, out_C, kernel_size=(1,1)))
            if dim == 1:
                in_C += 64

        self.right_path = torch.nn.Sequential()
        self.expected_C_params += 1
        self.expected_K_params += 2
        self.right_path.add_module("conv_b", torch.nn.Conv2d(64, 64*ratio, kernel_size=(1,1)))

        self.out_conv = torch.nn.Sequential()
        self.expected_C_params += 1
        if dim == 1:
            out_C += 64*ratio
        self.out_conv.add_module("conv_out", torch.nn.Conv2d(out_C, 16, kernel_size=(1,1)))

    def forward(self, input: torch.Tensor):
        step0 = self.in_conv(input)
        step1 = step0
        for d, layer in enumerate(self.left_paths):
            if d == 0:
                step1 = layer(step1)
            else:
                step1 = layer(torch.cat([step1, step0], 1))

        step2 = torch.cat([step1, self.right_path(step0)], self.dim)
        return self.out_conv(step2)

class test_flatten_op(torch.nn.Module):
    """flatten ops may change the effective channel count, typically by collapsing N,C,H,W into N,C*H*W before a classifier"""

    def __init__(
        self,
        change_dims = True,
    ):
        super().__init__()
        self.change_dims = change_dims
        self.input_shape = [4, 16, 3, 3]
        self.expected_C_params = 0
        self.expected_K_params = 0

        if not self.change_dims:
            self.input_shape = [4, 16, 1, 1]
            self.expected_C_params = 1
            self.expected_K_params = 2

        self.flattened_C = self.input_shape[2] * self.input_shape[3] * 64

        self.in_conv = torch.nn.Conv2d(self.input_shape[1], 64, kernel_size=(1,1))
        self.out_gemm = torch.nn.Linear(self.flattened_C, 16)


    def forward(self, input: torch.Tensor):
        step0 = self.in_conv(input)
        step1 = torch.flatten(step0, start_dim=1)
        return self.out_gemm(step1)

class test_flatten_module(torch.nn.Module):
    """flatten modules may change the effective channel count, typically by collapsing N,C,H,W into N,C*H*W before a classifier"""

    def __init__(
        self,
        change_dims = True,
    ):
        super().__init__()
        self.change_dims = change_dims
        self.input_shape = [4, 16, 3, 3]
        self.expected_C_params = 0
        self.expected_K_params = 0

        if not self.change_dims:
            self.input_shape = [4, 16, 1, 1]
            self.expected_C_params = 1
            self.expected_K_params = 2

        self.flattened_C = self.input_shape[2] * self.input_shape[3] * 64
        self.stack = torch.nn.Sequential()
        self.stack.add_module("conv_in", torch.nn.Conv2d(self.input_shape[1], 64, kernel_size=(1,1)))
        self.stack.add_module("flatten", torch.nn.Flatten(1))
        self.stack.add_module("gemm_out", torch.nn.Linear(self.flattened_C, 16))

    def forward(self, input: torch.Tensor):
        return self.stack(input)

class test_trace_failure(torch.nn.Module):
    """make sure tracing failures are handled gracefully"""

    def __init__(
        self
    ):
        super().__init__()
        self.input_shape = [4, 16, 1, 1]
        self.expected_C_params = 0
        self.expected_K_params = 0

        self.in_conv = torch.nn.Conv2d(self.input_shape[1], 64, kernel_size=(1,1))
        self.out_conv = torch.nn.Conv2d(64, 16, kernel_size=(1,1))

    def forward(self, input: torch.Tensor):
        step0 = self.in_conv(input)
        #NCHW = 4,64,1,1
        channels = step0.size(1)
        channel_offset = torch.arange(channels, dtype=torch.long, device=step0.device)
        channel_offset = channel_offset.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(step0)
        step0.add_(channel_offset)
        return self.out_conv(step0)

class already_sparse(torch.nn.Module):
    """if weights are already sparse, permutations should be skipped"""

    def __init__(
        self
    ):
        super().__init__()
        self.input_shape = [4, 16, 3, 3]
        self.expected_C_params = 0
        self.expected_K_params = 0

        self.in_conv = torch.nn.Conv2d(self.input_shape[1], 64, kernel_size=(1,1))
        self.out_conv = torch.nn.Conv2d(64, 16, kernel_size=(1,1))

        # apply 2:4 to the output weights, it will not require a permutation
        out_weights = torch.ones_like(self.out_conv.weight)
        out_weights[:,0::2,...] = 0
        assert torch.sum(out_weights) == torch.numel(out_weights)/2
        self.out_conv.weight.data.copy_(out_weights)

    def forward(self, input: torch.Tensor):
        step0 = self.in_conv(input)
        return self.out_conv(step0)

def test_model(model, tag, verbosity=0, save_onnx=False):
    Permutation.set_identical_seed()
    x = torch.rand(model.input_shape)
    if save_onnx:
        torch.onnx.export(model, x, f"{tag}.onnx", verbose=False)

    base_out = model(x)

    sparse_parameters = []
    all_parameters = []

    module_to_params = {}
    module_to_params[torch.nn.MultiheadAttention] = ('q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight')

    for module_name, module in model.named_modules():
        module_type_str = str(type(module)).split("\'")[1]
        if module_type_str == 'torch.nn.modules.container.Sequential' or module_type_str.startswith('torchvision.models'):
            # filter out the 'torch.nn.modules.container.Sequential' type and the whole model, like 'torchvision.models.vgg.VGG'
            continue
        for p_name, p in module.named_parameters():
            all_parameters.append((module_name, module, p_name, p))

            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.MultiheadAttention, torch.nn.modules.linear.NonDynamicallyQuantizableLinear)):
                allowed_names = ('weight',)
                if type(module) in module_to_params.keys():
                    allowed_names = module_to_params[type(module)]

                if p_name not in allowed_names:
                    continue

                if len(p.size()) >= 2 and (p.size()[0] % 8) == 0 and (p.size()[1] % 16) == 0:
                    mask = torch.ones_like(p).bool()
                    buffname = p_name.split(".")[-1]
                    module.register_buffer('__%s_mma_mask' % buffname, mask)
                    sparse_parameters.append((module_name, module, p_name, p, mask, None))

        if module_type_str == 'torch.nn.modules.batchnorm.BatchNorm2d':
        # need to get the running_mean and running_var from model.state_dict(), as they are not the learnable parameters
            module_mean_name = module_name + '.running_mean'
            module_var_name = module_name + '.running_var'
            for param_key in model.state_dict():
                if module_mean_name == param_key or module_var_name == param_key:
                    all_parameters.append((module_name, module, param_key.split(".")[-1], model.state_dict()[param_key]))

    if verbosity > 1:
       sparse_param_names = [module_name+":"+p_name for (module_name, module, p_name, p, mask, pruned) in sparse_parameters]
       all_param_names = [module_name+":"+p_name for (module_name, module, p_name, p) in all_parameters]
       print(f"\tSparse parameter names: {sparse_param_names}\n\tAll parameter names: {all_param_names}")

    Permutation.set_permutation_params_from_asp(model, sparse_parameters, all_parameters, verbosity)
    Permutation.permute_model(model)

    C_params, K_params, missed_dims = Permutation.get_permutation_stats()

    success = True
    fail_str = ""
    succ_str = ""
    if len(C_params) != model.expected_C_params:
        success = False
        fail_str = fail_str + f"\n\tC expected {model.expected_C_params}, got {len(C_params)} ({C_params})"
    elif verbosity > 0:
        succ_str = succ_str + f"\n\tC expected {model.expected_C_params}, got {len(C_params)} ({C_params})"
    
    if len(K_params) != model.expected_K_params:
        success = False
        fail_str = fail_str + f"\n\tK expected {model.expected_K_params}, got {len(K_params)} ({K_params})"
    elif verbosity > 0:
        succ_str = succ_str + f"\n\tK expected {model.expected_K_params}, got {len(K_params)} ({K_params})"
    
    if len(missed_dims) != 0:
        success = False
        fail_str = fail_str + f"\n\tMissed permutations along {len(missed_dims)} dimensions ({missed_dims})"

    perm_out = model(x)

    atol = 1e-5
    rtol = 1e-4
    outs_match = torch.allclose(base_out.data, perm_out.data, atol=atol, rtol=rtol)
    if not outs_match:
        fail_str = fail_str + f"\n\tOutputs matched: {outs_match}"
        if success:
            diffs = base_out - perm_out
            diff_locs = (diffs >= atol).nonzero(as_tuple=True)
            fail_str = fail_str + f"\n{diff_locs}\n{diffs[diff_locs]}"
        success = False

    if success:
        print(f"{tag}: Success\t{succ_str}")
    else:
        print(f"{tag}: FAIL\t{fail_str}")

    return success

def main():
 
    global_success = True

    global_success &= test_model(simple_convs(2,16), "smoke test")
    global_success &= test_model(simple_convs(5, 64), "simple 5 64")
    global_success &= test_model(simple_convs(10, 32), "simple 10 32")
    # normalization
    for norm in ['BatchNorm2d', 'LazyBatchNorm2d', 'InstanceNorm2d', 'LazyInstanceNorm2d', 'LayerNorm3', 'LocalResponseNorm']:
        global_success &= test_model(simple_convs(4, 128, norm), norm)
    # disallowed normalization
    for norm in ['GroupNorm']:
        global_success &= test_model(simple_convs(4, 128, norm), norm)
    
    global_success &= test_model(conv_1d(), "conv1d")
    global_success &= test_model(conv_1d(with_2d=True), "conv1d and conv2d")
    global_success &= test_model(grouped_convs(), "grouped convs")
    global_success &= test_model(simple_forks_joins(), "forks and joins")
    global_success &= test_model(different_grouped_convs(), "GCD")
    global_success &= test_model(siblings_poison(), "sibling poison")
    global_success &= test_model(coparent_poison(), "coparent poison")
    global_success &= test_model(depthwise_child_is_sibling(), "dw child is sibling")
    global_success &= test_model(module_attribute(complexity=0), "single attribute")
    global_success &= test_model(module_attribute(complexity=1), "single attribute thrice")
    global_success &= test_model(MHA_test(hidden_dim=256, seq_len=64, num_heads=16), "stacked MHA")
    global_success &= test_model(one_sparse_sibling(), "one sparse sibling")
    global_success &= test_model(test_concat(), "simple concat")  # concat along K
    global_success &= test_model(test_concat(dim=0), "concat dim0")  # concat along C
    global_success &= test_model(test_concat(ratio=2), "concat ratio2")  # concat along K with different K values
    global_success &= test_model(test_concat(depth=2), "concat depth2")  # concat along K multiple times
    global_success &= test_model(test_concat(depth=3), "concat depth3")
    global_success &= test_model(test_concat(ratio=3, depth=4), "concat ratio3 depth4")
    global_success &= test_model(test_concat(dim=0, depth=3), "concat dim0 depth3")
    global_success &= test_model(test_flatten_op(), "flatten op")
    global_success &= test_model(test_flatten_op(change_dims=False), "useless flatten op")
    global_success &= test_model(test_flatten_module(), "flatten module")
    global_success &= test_model(test_flatten_module(change_dims=False), "useless flatten module")
    global_success &= test_model(test_trace_failure(), "trace failure")
    global_success &= test_model(already_sparse(), "skip already sparse")
    global_success &= test_model(square_attribute(), "square attributes")

    if global_success:
        print("All tests completed successfully.")
    else:
        print("There was at least one failure.")

if __name__ == '__main__':
    main()
