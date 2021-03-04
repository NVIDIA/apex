import torch
import unittest
import numpy as np

import torch.nn.functional as F

from apex.contrib.layer_norm import FastLayerNorm

import fast_layer_norm as fln


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
def abs_err(x, y):
    xf = x.float()
    yf = y.float()
    return ((xf-yf).abs().sum() / yf.abs().sum()).item()



class TestFastLayerNorm(unittest.TestCase):
    
    def setUp(self, seed=1234):
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def test_ln_fp32(self):
        self.run_test_layer_norm(torch.float32, atol=1e-5)
    def test_ln_fp16(self):
        self.run_test_layer_norm(torch.float16, atol=1e-2, rtol=1e-3)

    def run_test_layer_norm(self, dtype, atol, rtol=1e-5):
        device = torch.device('cuda')
        s = 512
        b = 32
        hidden_size = 1024
        epsilon = 1e-5

        x = torch.randn((s,b,hidden_size), dtype=dtype, device=device)  
        beta = torch.randn(hidden_size, dtype=dtype, device=device)  
        gamma = torch.randn(hidden_size, dtype=dtype, device=device)
        x.requires_grad = True
        beta.requires_grad = True
        gamma.requires_grad = True

        x2 = x.clone().detach()
        beta2 = beta.clone().detach()
        gamma2 = gamma.clone().detach()
        x2.requires_grad = True
        beta2.requires_grad = True
        gamma2.requires_grad = True
               
        dummy_label = torch.randn_like(x)

        y = F.layer_norm(x, [hidden_size], gamma, beta, epsilon)

        diff = y-dummy_label
        l = (diff * diff).sum() / b
        l.backward()

        fln = FastLayerNorm(hidden_size).cuda()
        fln.load_state_dict({'bias': beta2, 'weight':gamma2})
        if dtype == torch.float16:
            fln = fln.half()

        y2 = fln(x2)
        diff2 = (y2 - dummy_label)
        l2 = (diff2 * diff2).sum() / b

        l2.backward()

        self.assertTrue(torch.allclose(y2, y, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(x2.grad, x.grad, atol=atol,rtol=rtol))
        self.assertTrue(torch.allclose(fln.bias.grad, beta.grad, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(fln.weight.grad, gamma.grad, atol=atol, rtol=rtol))
    


    def test_performance(self):
        print()
        runs = 1000
        device = torch.device('cuda')
        dtype =torch.float16
        s = 512
        b = 32
        hidden_size = 1024
        epsilon = 1e-5

        x = torch.randn((s*b,hidden_size), dtype=dtype, device=device)  
        beta = torch.randn(hidden_size, dtype=dtype, device=device)  
        gamma = torch.randn(hidden_size, dtype=dtype, device=device)
        dy = torch.randn_like(x)
 

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):

            timer = GPUTimer(stream)

            #warmup
            for r in range(runs):
                y, mu, rsigma = fln.ln_fwd(x, gamma, beta, 1e-5)
           
           
            timer.start()
            for r in range(runs):
                y, mu, rsigma = fln.ln_fwd(x, gamma, beta, 1e-5)
            timer.stop()
            timer.sync()

            total_bytes_fwd = (size_in_bytes(x) 
                             + size_in_bytes(y) 
                             + size_in_bytes(gamma) 
                             + size_in_bytes(beta) 
                             + size_in_bytes(mu) 
                             + size_in_bytes(rsigma)
                             )

            ms_fwd = timer.millis() / runs
            print('[FWD] Time: {:.4f}ms Throughput: {:.4f} GB/sec'.format(ms_fwd, total_bytes_fwd * 1e-6 / ms_fwd ))
         

            timer.start()
            for r in range(runs):
                dx, dgamma, dbeta = fln.ln_bwd(dy, x, mu, rsigma, gamma)
            timer.stop()
            timer.sync()

            total_bytes_bwd = (size_in_bytes(x) 
                             + size_in_bytes(dx)
                             + size_in_bytes(dy) 
                             + size_in_bytes(gamma) 
                             + size_in_bytes(dgamma)  
                             + size_in_bytes(dbeta)  
                             + size_in_bytes(mu) 
                             + size_in_bytes(rsigma)
                             )


            ms_bwd = timer.millis() / runs
            print('[BWD] Time: {:.4f}ms Throughput: {:.4f} GB/sec'.format(ms_bwd, total_bytes_bwd * 1e-6 / ms_bwd ))

if __name__ == '__main__':
    unittest.main()
