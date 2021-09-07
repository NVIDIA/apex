import torch
import unittest
import numpy as np
import random
from apex.contrib.groupbn.batch_norm import BatchNorm2d_NHWC

def generate_uniform_tensor(size, np_dtype, pyt_dtype, device):
    array = None
    while array is None or np.isnan(array).any():
        array = np.random.uniform(low=-1.0, high=1.0, size=size).astype(np_dtype)
    return torch.from_numpy(array).to(device).to(pyt_dtype)

def to_channels_last(tensor):
    return tensor.permute(0, 2, 3, 1).contiguous()

def to_channels_first(tensor):
    return tensor.permute(0, 3, 1, 2).contiguous()

class Bn(torch.nn.BatchNorm2d):
    def __init__(self, planes, mode):
        super(Bn, self).__init__(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mode = mode

    def forward(self, x, z=None):
        out = super().forward(x)
        if self.mode == 'bn_add_relu':
            out = out.add_(z)
        if self.mode != 'bn':
            out = out.relu_()
        return out

def bn_nhwc_bwd_ref(grad_y, x, mu, ivar, gamma):
    sum_dim_c = (0, 1, 2)
    grad_y_f32 = grad_y.float()
    x_f32 = x.float()
    N = x.shape[0] * x.shape[1] * x.shape[2] # nhw
    ones = torch.ones(x.shape, dtype=torch.float32, device='cuda')

    xmu = x_f32 - mu
    xhat = xmu * ivar

    dbias = torch.sum(grad_y_f32, dim=sum_dim_c)

    dscale = torch.sum(grad_y_f32 * xhat, dim=sum_dim_c)

    dx1 = (gamma * ivar) / N
    dx2 = (N * grad_y_f32) - (dbias * ones)
    dx3 = -xhat * dscale
    dx = dx1 * (dx2 + dx3)
    dx = dx.half()
    return dx, dscale, dbias

class TestGroupBN(unittest.TestCase):

    def setUp(self, seed=5, verbose=False):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.verbose = verbose

    def test_bn(self):
        self.run_group_bn('bn')

    def test_bn_relu(self):
        self.run_group_bn('bn_relu')

    def test_bn_add_relu(self):
        self.run_group_bn('bn_add_relu')

    def run_group_bn(self, mode):
        if self.verbose:
            print('Running {}'.format(mode))

        tensor_sizes = [
                (120, 64, 150, 150),
                (120, 64, 75, 75),
                (120, 128, 38, 38),
                (120, 256, 38, 38)]

        for i in range(len(tensor_sizes)):
            tensor_size = tensor_sizes[i]
            num_channels = tensor_size[1]

            # Create input data
            input_data = generate_uniform_tensor(tensor_size, np.float16, torch.half, 'cuda')
            np.save('input.npy', input_data.detach().cpu().numpy())
            input_data.requires_grad = True

            gbn_input = torch.from_numpy(np.load('input.npy')).cuda().half()
            gbn_input.requires_grad = True

            residual_data = None
            gbn_residual_data = None
            if mode == 'bn':
                fuse_relu = False
            else:
                fuse_relu = True
                if mode == 'bn_add_relu':
                    residual_data = generate_uniform_tensor(tensor_size, np.float16, torch.half, 'cuda')
                    gbn_residual_data = to_channels_last(residual_data)

            bn_grad = generate_uniform_tensor(input_data.shape, np.float16, torch.half, 'cuda')

            # Create models
            batchnorm_model = Bn(num_channels, mode).cuda()
            group_batchnorm = BatchNorm2d_NHWC(num_channels, fuse_relu=fuse_relu, bn_group=1).cuda()

            # Run reference forward
            bn_output = batchnorm_model(input_data, residual_data)

            # Run GBN forward
            gbn_input_data = to_channels_last(gbn_input)
            gbn_output = group_batchnorm(gbn_input_data, gbn_residual_data)

            torch.cuda.synchronize()

            # Run reference backward
            # (Use the same input and parameters as GBN)
            gbn_grad = to_channels_last(bn_grad)
            grad = gbn_grad.clone().detach()
            input_data = torch.from_numpy(np.load('input.npy')).cuda().half()
            input_data = to_channels_last(input_data)
            if mode != 'bn':
                grad[gbn_output <= 0] = 0
            bn_output_grad, _, _ = bn_nhwc_bwd_ref( \
                    grad,
                    input_data,
                    group_batchnorm.minibatch_mean,
                    group_batchnorm.minibatch_riv,
                    group_batchnorm.weight)
            bn_output_grad = to_channels_first(bn_output_grad)

            # Run GBN backward
            gbn_output.backward(gbn_grad)
            torch.cuda.synchronize()

            gbn_output = to_channels_first(gbn_output)
            gbn_output_grad = gbn_input.grad.detach().clone().cpu()

            ########################## Validate results ##########################
            if self.verbose:
                print('Validate activation')
            self.validate(bn_output.shape, bn_output, gbn_output)
            if self.verbose:
                print('Validate grad')
            self.validate(bn_output_grad.shape, bn_output_grad, gbn_output_grad, is_grad=True)

    def validate(self, tensors, output_ref, output_test, is_grad=False):
        output_ref = output_ref.detach().cpu().numpy()
        output_test = output_test.detach().cpu().numpy()

        if self.verbose:
            print('>>> tensor_size\t{}'.format(tensors))
            print("sum_output_ref {}, isnan {}, max {}, min {}".format(
                np.sum(output_ref, dtype=float), np.isnan(output_ref).any(), np.max(output_ref), np.min(output_ref)))
            print("sum_output_test {}, isnan {}, max {}, min {}".format(
                np.sum(output_test, dtype=float), np.isnan(output_test).any(), np.max(output_test), np.min(output_test)))

        ret = np.array_equal(output_ref, output_test)
        if not ret:
            ret_allclose = np.allclose(
                    output_ref, output_test, rtol=1e-3, atol=1e-3, equal_nan=True)
            if self.verbose:
                print('{}\tshape {}\tidentical {}\tclose {}'.format('cpu/gpu', tensors, ret, ret_allclose))
            output_ref = output_ref.flatten()
            output_test = output_test.flatten()
            if not ret:
                sub = np.absolute(output_ref - output_test)
                norm_diff = np.average(sub)
                rel = np.divide(sub, np.absolute(output_ref))
                rel[rel == np.inf] = 0
                max_abs_idx = np.argmax(sub)
                max_rel_idx = np.argmax(rel)
                if self.verbose:
                    print('max_diff {}, max_rel_diff {}, norm_diff {}'.format(np.max(sub), np.max(rel), np.average(sub)))
                    print('max_abs pair [{}] {} {}'.format(max_abs_idx, output_ref[max_abs_idx], output_test[max_abs_idx]))
                    print('max_rel pair [{}] {} {}'.format(max_rel_idx, output_ref[max_rel_idx], output_test[max_rel_idx]))

        result = ret or ret_allclose or (is_grad and norm_diff < 1e-4)

        if self.verbose:
            print("Result {}".format("PASS" if result else "FAIL"))

        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
