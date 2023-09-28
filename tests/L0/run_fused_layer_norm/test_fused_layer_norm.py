import torch
from apex.normalization import FusedLayerNorm
from apex.normalization import FusedRMSNorm
from apex.normalization import MixedFusedLayerNorm
from apex.normalization import MixedFusedRMSNorm

from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests

from itertools import product

def _prep_inputs(batch_size, normalized_shape, dtype):
    shape = (batch_size, *normalized_shape)
    fused = torch.randn(shape).cuda().requires_grad_(True)
    with torch.no_grad():
        native = fused.clone().to(dtype).requires_grad_(True)
    return native, fused

autocast_dtypes = (torch.half, torch.bfloat16) if torch.cuda.is_bf16_supported() else (torch.half,)

class TestFusedLayerNorm(common_utils.TestCase):

    def _test_fused_layer_norm(
        self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient,
        fwd_thresholds=dict(rtol=None, atol=None), bwd_thresholds=dict(rtol=None, atol=None)
        ):

        normalized_shape = [32, 16]

        if not mixed_fused:
            module_cpu_ = FusedLayerNorm(
                normalized_shape=normalized_shape, elementwise_affine=elementwise_affine, memory_efficient=memory_efficient
            ).cpu()
            module_cuda_ = FusedLayerNorm(
                normalized_shape=normalized_shape, elementwise_affine=elementwise_affine, memory_efficient=memory_efficient
            ).to(device="cuda", dtype=dtype)
        else:
            assert elementwise_affine
            module_cpu_ = MixedFusedLayerNorm(
                normalized_shape=normalized_shape, memory_efficient=memory_efficient
            ).cpu()
            module_cuda_ = MixedFusedLayerNorm(
                normalized_shape=normalized_shape, memory_efficient=memory_efficient
            ).to(device="cuda", dtype=dtype)

        torch.cuda.manual_seed(42)
        if contiguous:
            input_shape = [batch_size] + normalized_shape
            input_ = torch.randn(input_shape, device="cpu").requires_grad_(True)
            input_cuda_ = input_.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
            self.assertTrue(input_.is_contiguous())
            self.assertTrue(input_cuda_.is_contiguous())
        else:
            input_shape = [batch_size] + normalized_shape
            input_shape = [batch_size * 3] + [normalized_shape[0] * 5, normalized_shape[1] * 3]
            input_src_ = torch.randn(input_shape, device="cpu")
            input_ = input_src_[::3, ::5, ::3].detach().requires_grad_(True)
            input_cuda_ = input_src_.to(device="cuda", dtype=dtype)[::3, ::5, ::3].detach().requires_grad_(True)
            # make sure that tensors are NOT contiguous.
            self.assertFalse(input_.is_contiguous())
            self.assertFalse(input_cuda_.is_contiguous())
        out_cpu_ = module_cpu_(input_)
        gO = torch.rand_like(out_cpu_)
        out_cpu_.backward(gO)
        out_cuda_ = module_cuda_(input_cuda_)

        gO = gO.to(device="cuda", dtype=dtype)
        out_cuda_.backward(gO)
        self.assertFalse(out_cpu_.is_cuda)
        self.assertTrue(out_cuda_.is_cuda)
        torch.testing.assert_close(
            out_cpu_.to(device="cuda", dtype=dtype), out_cuda_, **fwd_thresholds)
        torch.testing.assert_close(
            input_.grad.to(device="cuda", dtype=dtype), input_cuda_.grad, **bwd_thresholds)

    def _test_fused_rms_norm(
        self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient,
        fwd_thresholds=dict(rtol=None, atol=None), bwd_thresholds=dict(rtol=None, atol=None)
        ):

        normalized_shape = [32, 16]

        if not mixed_fused:
            module_cpu_ = FusedRMSNorm(
                normalized_shape=normalized_shape, elementwise_affine=elementwise_affine, memory_efficient=memory_efficient
            ).cpu()
            module_cuda_ = FusedRMSNorm(
                normalized_shape=normalized_shape, elementwise_affine=elementwise_affine, memory_efficient=memory_efficient
            ).to(device="cuda", dtype=dtype)
        else:
            assert elementwise_affine
            module_cpu_ = MixedFusedRMSNorm(
                normalized_shape=normalized_shape).cpu()
            module_cuda_ = MixedFusedRMSNorm(
                normalized_shape=normalized_shape).to(device="cuda", dtype=dtype)

        torch.cuda.manual_seed(42)
        if contiguous:
            input_shape = [batch_size] + normalized_shape
            input_ = torch.randn(input_shape, device="cpu").requires_grad_(True)
            input_cuda_ = input_.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
            self.assertTrue(input_.is_contiguous())
            self.assertTrue(input_cuda_.is_contiguous())
        else:
            input_shape = [batch_size] + normalized_shape
            input_shape = [batch_size * 3] + [normalized_shape[0] * 5, normalized_shape[1] * 3]
            input_src_ = torch.randn(input_shape, device="cpu")
            input_ = input_src_[::3, ::5, ::3].detach().requires_grad_(True)
            input_cuda_ = input_src_.to(device="cuda", dtype=dtype)[::3, ::5, ::3].detach().requires_grad_(True)
            # make sure that tensors are NOT contiguous.
            self.assertFalse(input_.is_contiguous())
            self.assertFalse(input_cuda_.is_contiguous())
        out_cpu_ = module_cpu_(input_)
        gO = torch.rand_like(out_cpu_)
        out_cpu_.backward(gO)
        out_cuda_ = module_cuda_(input_cuda_)

        torch.testing.assert_close(
            out_cpu_.to(device="cuda", dtype=dtype), out_cuda_.clone().detach(), **fwd_thresholds)
        gO = gO.to(device="cuda", dtype=dtype)
        out_cuda_.backward(gO)
        self.assertFalse(out_cpu_.is_cuda)
        self.assertTrue(out_cuda_.is_cuda)
        torch.testing.assert_close(
            input_.grad.to(device="cuda", dtype=dtype), input_cuda_.grad, **bwd_thresholds)
        if elementwise_affine:
            torch.testing.assert_close(module_cpu_.weight.grad.to(device="cuda", dtype=dtype),
                                          module_cuda_.weight.grad, **bwd_thresholds)

    # layer norm tests
    @common_utils.parametrize(
        "batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient",
        list(product((16, 65536), (True, False), (False,), (False,), (torch.float,), (True, False)))
    )
    def test_layer_norm_regular(self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient):
        self._test_fused_layer_norm(batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient)
    
    @common_utils.parametrize(
        "batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient",
        list(product((16, 65536), (True, False), (True,), (False,), (torch.float,), (True, False)))
    )
    def test_layer_norm_elemwise(self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient):
        self._test_fused_layer_norm(batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient)

    @common_utils.parametrize(
        "batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient",
        list(product((16, 65536), (True, False), (True,), (True,), (torch.float,), (True, False)))
    )
    def test_layer_norm_mixed(self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient):
        self._test_fused_layer_norm(batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient)
    
    @common_utils.parametrize(
        "batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient",
        list(product((16,), (True, False), (True,), (False,), (torch.half,), (True, False)))
    )
    def test_layer_norm_half(self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient):
        self._test_fused_layer_norm(batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient,
                                fwd_thresholds=dict(rtol=1e-3, atol=1e-3), bwd_thresholds=dict(rtol=1e-3, atol=1e-3))
    
    @common_utils.parametrize(
        "batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient",
        list(product((16,), (True, False), (True,), (False,), (torch.bfloat16,), (True, False)))
    )
    def test_layer_norm_bfloat16(self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient):
        self._test_fused_layer_norm(batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient, 
                                fwd_thresholds=dict(rtol=1.6e-2, atol=3e-4), bwd_thresholds=dict(rtol=1.6e-2, atol=3e-3))

    # rms norm tests
    @common_utils.parametrize(
        "batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient",
        list(product((16, 65536), (True, False), (False,), (False,), (torch.float,), (True, False)))
    )
    def test_rms_norm_regular(self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient):
        self._test_fused_rms_norm(batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient)

    @common_utils.parametrize(
        "batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient",
        list(product((16, 65536), (True, False), (True,), (False,), (torch.float,), (True, False)))
    )
    def test_rms_norm_elemwise(self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient):
        self._test_fused_rms_norm(batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient,
                                bwd_thresholds=dict(rtol=2e-3, atol=2e-4))

    @common_utils.parametrize(
        "batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient",
        list(product((16, 65536), (True, False), (True,), (True,), (torch.float,), (True, False)))
    )
    def test_rms_norm_mixed(self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient):
        self._test_fused_rms_norm(batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient,
                                bwd_thresholds=dict(rtol=2e-3, atol=2e-4))
    
    @common_utils.parametrize(
        "batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient",
        list(product((16,), (True, False), (True,), (False,), (torch.half,), (True, False)))
    )
    def test_rms_norm_half(self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient):
        self._test_fused_rms_norm(batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient,
                                bwd_thresholds = dict(rtol=1.6e-2, atol=3e-3))
    
    @common_utils.parametrize(
        "batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient",
        list(product((16,), (True, False), (True,), (False,), (torch.bfloat16,), (True, False)))
    )
    def test_rms_norm_bfloat16(self, batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient):
        self._test_fused_rms_norm(batch_size, contiguous, elementwise_affine, mixed_fused, dtype, memory_efficient, 
                                fwd_thresholds=dict(rtol=1.6e-2, atol=3e-4), bwd_thresholds=dict(rtol=1.6e-2, atol=3e-2))

    @common_utils.parametrize(
        "dtype, elementwise_affine, memory_efficient",
        list(product(autocast_dtypes, (True, False), (True, False)))
    )
    def test_autocast_fused_layer_norm(self, dtype, elementwise_affine, memory_efficient):
        bf16_fwd_thresholds = dict(rtol=1.6e-2, atol=3e-4)
        bf16_bwd_thresholds = dict(rtol=1.6e-2, atol=3e-3)
        batch_size = 16
        normalized_shape = [32, 16]
        native = torch.nn.LayerNorm(
            normalized_shape=normalized_shape, elementwise_affine=elementwise_affine
        ).to(device="cuda", dtype=dtype)
        fused = FusedLayerNorm(
            normalized_shape=normalized_shape, elementwise_affine=elementwise_affine, memory_efficient=memory_efficient
        ).cuda()
        native_x, fused_x = _prep_inputs(batch_size, normalized_shape, dtype)

        expected = native(native_x)
        with torch.cuda.amp.autocast(dtype=dtype):
            actual = fused(fused_x)
        tols = {'rtol': None, 'atol': None} if dtype == torch.half else bf16_fwd_thresholds
        # original tests used torch.testing.assert_allclose, which disables dtype checking by default. 
        # link to issue here: https://github.com/pytorch/pytorch/issues/61844
        torch.testing.assert_close(actual, expected, **tols, check_dtype=False) 

        g_native = torch.rand_like(expected)
        with torch.no_grad():
            g_fused = g_native.clone()
        expected.backward(g_native)
        actual.backward(g_fused)

        if dtype != torch.half:
            tols = bf16_bwd_thresholds
        elif memory_efficient:
            tols = {'rtol': 1e-3, 'atol': 1e-4}
        else:
            tols = {'rtol': None, 'atol': None}
        torch.testing.assert_close(native_x.grad, fused_x.grad, **tols, check_dtype=False)
    @common_utils.parametrize(
        "dtype, elementwise_affine, memory_efficient",
        list(product(autocast_dtypes, (True, False), (True, False)))
    )
    def test_autocast_fused_rms_norm(self, dtype, elementwise_affine, memory_efficient):
        bf16_fwd_thresholds = dict(rtol=1.6e-2, atol=3e-4)
        bf16_bwd_thresholds = dict(rtol=1.6e-2, atol=3e-3)
        batch_size = 16
        normalized_shape = [32, 16]
        native = FusedRMSNorm(
            normalized_shape=normalized_shape, elementwise_affine=elementwise_affine, memory_efficient=memory_efficient, 
        ).to(dtype=dtype)
        fused = FusedRMSNorm(
            normalized_shape=normalized_shape, elementwise_affine=elementwise_affine, memory_efficient=memory_efficient, 
        ).cuda()
        native_x, fused_x = _prep_inputs(batch_size, normalized_shape, dtype)

        expected = native(native_x.cpu())
        with torch.cuda.amp.autocast(dtype=dtype):
            actual = fused(fused_x)
        tols = {'rtol': None, 'atol': None} if dtype == torch.half else bf16_fwd_thresholds
        torch.testing.assert_close(actual, expected.detach().clone().cuda(), **tols, check_dtype=False)

        g_native = torch.rand_like(expected)
        with torch.no_grad():
            g_fused = g_native.detach().clone().cuda()
        expected.backward(g_native)
        actual.backward(g_fused)

        tols = {'rtol': 1e-3, 'atol': 1e-3} if dtype == torch.half else bf16_bwd_thresholds
        torch.testing.assert_close(native_x.grad.cuda(), fused_x.grad, **tols, check_dtype=False)

    def _verify_export(self, fused, fused_x):
        # check that export() is working
        onnx_str = torch.onnx.export_to_pretty_string(fused, (fused_x,),
                                                      input_names=['x_in'],
                                                      opset_version=18,
        )
        assert 'x_in' in onnx_str
        assert 'ReduceMean' in onnx_str or 'LayerNormalization' in onnx_str

    def test_rms_export(self):
        batch_size = 16
        normalized_shape = [32, 16]
        fused = FusedRMSNorm(
            normalized_shape=normalized_shape, elementwise_affine=True
        ).cuda()
        fused_m = MixedFusedRMSNorm(
            normalized_shape=normalized_shape
        ).cuda()
        native_x, fused_x = _prep_inputs(batch_size, normalized_shape, torch.float32)
        self._verify_export(fused, fused_x)
        self._verify_export(fused_m, fused_x)
        
    def test_layer_norm_export(self):
        batch_size = 16
        normalized_shape = [32, 16]
        fused = FusedLayerNorm(
            normalized_shape=normalized_shape, elementwise_affine=True
        ).cuda()
        fused_m = MixedFusedLayerNorm(
            normalized_shape=normalized_shape
        ).cuda()
        native_x, fused_x = _prep_inputs(batch_size, normalized_shape, torch.float32)
        self._verify_export(fused, fused_x)
        self._verify_export(fused_m, fused_x)
        
instantiate_device_type_tests(TestFusedLayerNorm, globals(), only_for=("cuda",))
if __name__ == "__main__":
    common_utils.run_tests()
