import torch

import unittest

from apex.contrib.multihead_attn import EncdecMultiheadAttn

class EncdecMultiheadAttnNormAddTest(unittest.TestCase):
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
                                             include_norm_add=True, 
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
                                             include_norm_add=True, 
                                             impl='fast')
        self.tst_layer.cuda().half()
        self.tst_layer.reset_parameters()
        
        self.tst_inputs_q = torch.randn(self.seq_length, self.sequences, self.hidden_dim, 
                                        dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
        self.tst_inputs_k = torch.randn(self.seq_length, self.sequences, self.hidden_dim, 
                                        dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)

    def test_encdec_multihead_attn_norm_add(self) :
        grads         = torch.randn_like(self.tst_inputs_q)
        
        for _ in range(5) :
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

if __name__ == '__main__':
    unittest.main()
