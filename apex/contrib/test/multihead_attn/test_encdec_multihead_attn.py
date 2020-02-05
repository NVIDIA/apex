import torch

import unittest

from apex.contrib.multihead_attn import EncdecMultiheadAttn

class EncdecMultiheadAttnTest(unittest.TestCase):
    def setUp(self, seed=1234):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.seq_length   = 80
        self.sequences    = 10
        self.hidden_dim   = 1024
        self.heads        = 16
        self.dropout_prob = 0.0

        self.ref_layer = EncdecMultiheadAttn(self.hidden_dim, 
                                             self.heads, 
                                             dropout=self.dropout_prob, 
                                             bias=False, 
                                             include_norm_add=False, 
                                             impl='default')
        self.ref_layer.cuda().half()
        self.ref_layer.reset_parameters()
        self.ref_inputs_q = torch.randn(self.seq_length, self.sequences, self.hidden_dim, 
                                        dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
        self.ref_inputs_k = torch.randn(self.seq_length, self.sequences, self.hidden_dim, 
                                        dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

        # Reset seed so parameters are identical
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        self.tst_layer = EncdecMultiheadAttn(self.hidden_dim, 
                                             self.heads, 
                                             dropout=self.dropout_prob, 
                                             bias=False, 
                                             include_norm_add=False, 
                                             impl='fast')
        self.tst_layer.cuda().half()
        self.tst_layer.reset_parameters()
        
        self.tst_inputs_q = torch.randn(self.seq_length, self.sequences, self.hidden_dim, 
                                        dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
        self.tst_inputs_k = torch.randn(self.seq_length, self.sequences, self.hidden_dim, 
                                        dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

    def test_encdec_multihead_attn(self) :
        grads         = torch.randn_like(self.tst_inputs_q)

        ref_outputs,_ = self.ref_layer.forward(self.ref_inputs_q, 
                                               self.ref_inputs_k, 
                                               self.ref_inputs_k,
                                               key_padding_mask=None, 
                                               need_weights=False, 
                                               attn_mask=None,
                                               is_training=True)

        tst_outputs,_ = self.tst_layer.forward(self.tst_inputs_q, 
                                               self.tst_inputs_k, 
                                               self.tst_inputs_k,
                                               key_padding_mask=None, 
                                               need_weights=False, 
                                               attn_mask=None,
                                               is_training=True)
        
        self.ref_inputs_q.backward(grads)
        self.tst_inputs_q.backward(grads)

        self.assertTrue(torch.allclose(self.ref_inputs_q,  self.tst_inputs_q,  atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(self.ref_inputs_k,  self.tst_inputs_k,  atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(self.ref_inputs_q.grad, self.tst_inputs_q.grad, atol=1e-3, rtol=1e-3))
    
    def test_encdec_multihead_attn_time_mask(self) :
        grads          = torch.randn_like(self.tst_inputs_q)
        time_mask_byte = torch.triu(torch.ones(self.tst_inputs_q.size(0), self.tst_inputs_k.size(0), device=torch.device("cuda"), dtype=torch.uint8), 1)
        time_mask_bool = time_mask_byte.to(torch.bool)
        
        ref_outputs,_ = self.ref_layer.forward(self.ref_inputs_q, 
                                               self.ref_inputs_k, 
                                               self.ref_inputs_k,
                                               key_padding_mask=None, 
                                               need_weights=False, 
                                               attn_mask=time_mask_bool,
                                               is_training=True)

        tst_outputs,_ = self.tst_layer.forward(self.tst_inputs_q, 
                                               self.tst_inputs_k, 
                                               self.tst_inputs_k,
                                               key_padding_mask=None, 
                                               need_weights=False, 
                                               attn_mask=time_mask_byte,
                                               is_training=True)
        
        self.ref_inputs_q.backward(grads)
        self.tst_inputs_q.backward(grads)

        self.assertTrue(torch.allclose(self.ref_inputs_q,  self.tst_inputs_q,  atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(self.ref_inputs_k,  self.tst_inputs_k,  atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(self.ref_inputs_q.grad, self.tst_inputs_q.grad, atol=1e-3, rtol=1e-3))
    
    def test_encdec_multihead_attn_pad_mask(self) :
        grads         = torch.randn_like(self.tst_inputs_q)
        pad_mask_byte = torch.tril(torch.ones(self.tst_inputs_k.size(1), self.tst_inputs_k.size(0), device=torch.device("cuda"), dtype=torch.uint8), 1)
        pad_mask_bool = pad_mask_byte.to(torch.bool)
        
        ref_outputs,_ = self.ref_layer.forward(self.ref_inputs_q, 
                                               self.ref_inputs_k, 
                                               self.ref_inputs_k,
                                               key_padding_mask=pad_mask_bool, 
                                               need_weights=False, 
                                               attn_mask=None,
                                               is_training=True)

        tst_outputs,_ = self.tst_layer.forward(self.tst_inputs_q, 
                                               self.tst_inputs_k, 
                                               self.tst_inputs_k,
                                               key_padding_mask=pad_mask_byte, 
                                               need_weights=False, 
                                               attn_mask=None,
                                               is_training=True)
        
        self.ref_inputs_q.backward(grads)
        self.tst_inputs_q.backward(grads)

        self.assertTrue(torch.allclose(self.ref_inputs_q,  self.tst_inputs_q,  atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(self.ref_inputs_k,  self.tst_inputs_k,  atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(ref_outputs, tst_outputs, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(self.ref_inputs_q.grad, self.tst_inputs_q.grad, atol=1e-3, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()
