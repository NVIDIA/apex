import unittest

import torch

SKIP_TEST = None
try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm
    import fast_layer_norm as fln
except ImportError as e:
    SKIP_TEST = e


class GPUTimer:
    def __init__(self, stream):
        self.start_ = torch.cuda.Event(enable_timing=True)
        self.stop_ = torch.cuda.Event(enable_timing=True)
        self.stream_ = stream

    def start(self):
        self.stream_.record_event(self.start_)

    def stop(self):
        self.stream_.record_event(self.stop_)

    def sync(self):
        self.stream_.synchronize()

    def millis(self):
        return self.start_.elapsed_time(self.stop_)


def size_in_bytes(t):
    return torch.numel(t) * t.element_size()


def metrics(y_ref, y, epsilon=1e-6):
    y_ref = y_ref.float()
    y = y.float()
    relerr, mse = (
        (y_ref - y).abs().sum() / (y_ref.abs().sum() + epsilon),
        (y_ref - y).square().mean(),
    )
    return relerr.item(), mse.item()


device = torch.device("cuda")
fp32 = torch.float32
fp16 = torch.float16
bf16 = torch.bfloat16


def backward_(dz, x, mu, rs, gamma):

    wtype = gamma.dtype
    itype = x.dtype
    otype = dz.dtype
    ctype = mu.dtype
    mu = mu.unsqueeze(1)
    rs = rs.unsqueeze(1)

    hidden_size = gamma.numel()
    y = rs * (x.to(ctype) - mu)
    dbeta = dz.view(-1, hidden_size).sum(0, dtype=ctype)
    dgamma = (dz * y).view(-1, hidden_size).sum(0, dtype=ctype)
    dy = dz.view(-1, hidden_size).to(ctype) * gamma.unsqueeze(0).to(ctype)
    mdy = dy.mean(1, keepdim=True, dtype=ctype)

    mdyy = (dy * y).mean(1, keepdim=True, dtype=ctype)
    dx = rs * (dy - mdyy * y - mdy)

    return dx.to(itype), dgamma.to(wtype), dbeta.to(wtype)


def benchmark_(S, B, hidden_size, itype, wtype, runs=100):
    epsilon = 1e-5

    x = torch.randn((S * B, hidden_size), dtype=itype, device=device)
    beta = torch.randn(hidden_size, dtype=wtype, device=device)
    gamma = torch.randn(hidden_size, dtype=wtype, device=device)
    dz = torch.randn(x.shape, dtype=wtype, device=device)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):

        timer = GPUTimer(stream)

        # warmup
        for r in range(runs):
            z, mu, rsigma = fln.ln_fwd(x, gamma, beta, epsilon)

        timer.start()
        for r in range(runs):
            z, mu, rsigma = fln.ln_fwd(x, gamma, beta, epsilon)
        timer.stop()
        timer.sync()

        total_bytes_fwd = sum([size_in_bytes(t) for t in [x, z, gamma, beta, mu, rsigma]])

        ms_fwd = timer.millis() / runs

        print(
            "[FWD] Time: {:.4f}ms Throughput: {:.4f} GB/sec".format(
                ms_fwd, total_bytes_fwd * 1e-6 / ms_fwd
            )
        )

        timer.start()
        for r in range(runs):
            dx, dgamma, dbeta, dbp, dgp = fln.ln_bwd(dz, x, mu, rsigma, gamma)
        timer.stop()
        timer.sync()

        total_bytes_bwd = sum(
            [
                size_in_bytes(t)
                for t in [dz, x, mu, rsigma, gamma, dx, dgamma, dbeta, dbp, dbp, dgp, dgp]
            ]
        )

        ms_bwd = timer.millis() / runs

        print(
            "[BWD] Time: {:.4f}ms Throughput: {:.4f} GB/sec".format(
                ms_bwd, total_bytes_bwd * 1e-6 / ms_bwd
            )
        )


def _test_impl(S, B, hidden_size, itype, wtype, ctype=fp32):

    seed = 1243
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    otype = wtype
    print("========================================================")
    print(f"S={S} B={B} Hidden={hidden_size} {itype} {wtype}")
    print("--------------------------------------------------------")

    x = torch.randn(S * B, hidden_size, dtype=itype, device=device)
    gamma = torch.randn(hidden_size, dtype=wtype, device=device) * 0.2
    beta = torch.randn(hidden_size, dtype=wtype, device=device) * 0.2
    epsilon = 1e-5

    x.requires_grad = True
    gamma.requires_grad = True
    beta.requires_grad = True

    mu_ref = x.mean(1, dtype=ctype, keepdim=True)
    v = torch.square(x - mu_ref).mean(1, dtype=ctype, keepdim=True)
    rs_ref = torch.rsqrt(v + epsilon)
    y_ref = rs_ref * (x.to(ctype) - mu_ref)
    z_ref = (gamma.unsqueeze(0) * (y_ref).to(otype) + beta.unsqueeze(0)).to(otype)

    mu_ref = mu_ref.flatten()
    rs_ref = rs_ref.flatten()

    dz = torch.randn_like(z_ref)

    # z_ref.backward(dz)
    # dx_ref = x.grad
    # dgamma_ref = gamma.grad
    # dbeta_ref = beta.grad

    dx_ref, dg_ref, db_ref = backward_(dz, x, mu_ref, rs_ref, gamma)

    z, mu, rs = fln.ln_fwd(x, gamma, beta, epsilon)
    dx, dg, db, dg_part, db_part = fln.ln_bwd(dz, x, mu, rs, gamma)

    re_z, mse_z = metrics(z_ref, z)
    re_mu, mse_mu = metrics(mu_ref, mu)
    re_rs, mse_rs = metrics(rs_ref, rs)

    re_dx, mse_dx = metrics(dx_ref, dx)
    re_dg, mse_dg = metrics(dg_ref, dg)
    re_db, mse_db = metrics(db_ref, db)

    print(f" z: relerr={re_z :.4e} mse={mse_z :.4e}")
    print(f"mu: relerr={re_mu:.4e} mse={mse_mu:.4e}")
    print(f"rs: relerr={re_mu:.4e} mse={mse_mu:.4e}")

    print(f"dx: relerr={re_dx:.4e} mse={mse_dx:.4e}")
    print(f"dg: relerr={re_dg:.4e} mse={mse_dg:.4e}")
    print(f"db: relerr={re_db:.4e} mse={mse_db:.4e}")

    def check_err(x, relerr):
        tol = 1e-3 if x.dtype == torch.float16 else 5e-6
        return relerr < tol

    return [
        check_err(x, re)
        for x, re in zip([z, mu, rs, dx, dg, db], [re_z, re_mu, re_rs, re_dx, re_dg, re_db])
    ]


@unittest.skipIf(SKIP_TEST, f"{SKIP_TEST}")
class TestFastLayerNorm(unittest.TestCase):
    # TODO(crcrpar): Try `torch.testing.assert_close` instead and migrate to it if it's working.
    def assertAll(self, l):
        if not all(l):
            print(l)
        for x in l:
            self.assertTrue(x)

    def test_all_configs(self):

        hidden_sizes = [
            768,
            1024,
            1536,
            2048,
            2304,
            3072,
            3840,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
            12800,
            14336,
            15360,
            16384,
            18432,
            20480,
            24576,
            25600,
            30720,
            32768,
            40960,
            49152,
            65536,
        ]

        for h in hidden_sizes:
            with self.subTest(f"hidden_size={h}"):
                self.assertAll(_test_impl(256, 2, h, fp32, fp32))
                self.assertAll(_test_impl(256, 2, h, fp16, fp16))
                self.assertAll(_test_impl(256, 2, h, fp32, fp16))
                self.assertAll(_test_impl(256, 2, h, bf16, bf16))
                self.assertAll(_test_impl(256, 2, h, fp32, bf16))

    def test_run_benchmark(self):
        for (S, B, hidden_size, runs) in (
            (512, 32, 768, 1000),
            (512, 32, 1024, 1000),
            (512, 8, 4096, 1000),
            (512, 8, 5120, 1000),
            (512, 8, 6144, 1000),
            (256, 2, 20480, 500),
            (256, 2, 25600, 500),
            (256, 2, 40960, 250),
            (256, 2, 65536, 250),
        ):
            with self.subTest(f"(S, B, hidden_size)=({S}, {B}, {hidden_size})"):
                benchmark_(S, B, hidden_size, fp16, fp16, runs)

    def test_compat_with_autocast(self):
        autocast_dtypes = (
            (torch.half, torch.bfloat16) if torch.cuda.is_bf16_supported() else (torch.half,)
        )
        input_shape = (512, 32, 768)
        layer_norm = FastLayerNorm(input_shape[-1]).cuda()
        input = torch.randn(input_shape).cuda()

        for dtype in autocast_dtypes:
            layer_norm.zero_grad(set_to_none=True)
            with self.subTest(f"autocast_dtype={dtype}"):
                with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                    out = layer_norm(input)
                    self.assertEqual(dtype, out.dtype)
                grad = torch.randn_like(out)
                out.backward(grad)
                self.assertEqual(torch.float32, layer_norm.weight.grad.dtype)


if __name__ == "__main__":
    unittest.main()
