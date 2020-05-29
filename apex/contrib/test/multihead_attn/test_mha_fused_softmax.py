import torch
import unittest
import torch.nn.functional as F
from apex.contrib.multihead_attn import fast_mask_softmax_dropout_func

class FusedSoftmaxTest(unittest.TestCase):
    def setUp(self, seed=1234):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.seq_length   = 80
        self.sequences    = 10
        self.hidden_dim   = 1024
        self.heads        = 16
        self.dropout_prob = 0.0

        self.mask = (torch.randn(self.sequences,self.seq_length)>0).cuda()
        self.mask = self.mask.half()*-10000
        self.ref_inputs = torch.randn(self.heads * self.sequences, self.seq_length, self.seq_length, 
                                      dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
        
        self.tst_inputs = self.ref_inputs.clone().detach().requires_grad_(True)

    def test_fused_softmax(self) :
        grads = torch.randn_like(self.tst_inputs)
        y_ref = self.ref_inputs.view(self.sequences, self.heads, self.seq_length, self.seq_length)
        y_ref = y_ref + self.mask.unsqueeze(1).unsqueeze(2)
        y_ref = y_ref.view(self.sequences*self.heads, self.seq_length, self.seq_length) 
        y_ref = F.softmax(y_ref, dim=-1)
        y_ref = torch._fused_dropout(y_ref, 1.0)    
   
        y_tst = fast_mask_softmax_dropout_func(True, self.heads, self.tst_inputs, self.mask, True, 0.0)        
        y_ref[0].backward(grads)
        y_tst.backward(grads)

        self.assertTrue(torch.allclose(self.ref_inputs,  self.tst_inputs,  atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(y_ref[0], y_tst, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(self.ref_inputs.grad, self.tst_inputs.grad, atol=1e-3, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()
