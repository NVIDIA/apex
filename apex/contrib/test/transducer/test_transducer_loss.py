import torch
import unittest
from apex.contrib.transducer import TransducerLoss
import transducer_ref

class TransducerLossTest(unittest.TestCase):
    def setUp(self, seed=1234):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def gen_input(self, scalar_t, for_vector_kernel):
        self.B = 5
        T_min = 23
        T_max = 51
        U_min = 12
        U_max = 25
        V = 16 if for_vector_kernel else 14
        self.blank_idx = V - 1
        device = "cuda"

        self.x_tst = torch.randn((self.B, T_max, U_max, V), dtype=scalar_t, requires_grad=True, 
                                    device=device)
        self.y = torch.randint(0, self.blank_idx, (self.B, U_max-1), dtype=torch.int, device=device)
        self.f_len = torch.randint(T_min, T_max+1, (self.B,), dtype=torch.int, device=device) 
        self.y_len = torch.randint(U_min-1, U_max, (self.B,), dtype=torch.int, device=device)
        self.f_len[torch.randint(0, self.B, (1,)).item()] = T_max
        self.y_len[torch.randint(0, self.B, (1,)).item()] = U_max-1
        self.x_tst_packed, self.batch_offset = self._pack(self.x_tst)
        # Generate reference
        x_ref = self.x_tst.data.clone()
        x_ref.requires_grad = True
        loss_grad = torch.ones(x_ref.size(0), dtype=x_ref.dtype, device=x_ref.device)/x_ref.size(0)
        _, _, self.grad_ref, self.loss_ref \
            = transducer_ref.transducer_loss_reference( x=x_ref, 
                                                        label=self.y, 
                                                        f_len=self.f_len, 
                                                        y_len=self.y_len, 
                                                        blank_idx=self.blank_idx, 
                                                        loss_grad=loss_grad)

    def _pack(self, x):
        list_x = []
        for b in range(self.B):
            list_x_row = [x[b, t, : self.y_len[b]+1] for t in range(self.f_len[b])]
            x_row = torch.cat(list_x_row)
            list_x.append(x_row)
        x_packed = torch.cat(list_x).data.clone()
        x_packed.requires_grad = True
        batch_offset = torch.cumsum(self.f_len * (self.y_len+1), dim=0)
        return x_packed, batch_offset

    def _unpack(self, x):
        x_unpacked = torch.zeros(self.B, self.f_len.max(), self.y_len.max()+1, x.size(-1), 
                                    dtype=x.dtype, device=x.device)
        for b in range(self.B):
            my_batch_offset = 0 if b == 0 else self.batch_offset[b-1]
            my_f_len = self.f_len[b]
            my_g_len = self.y_len[b] + 1
            for t in range(my_f_len):
                for u in range(my_g_len):
                    x_unpacked[b, t, u] = x[my_batch_offset + t*my_g_len + u]
        return x_unpacked

    def run_transducer_loss(self, scalar_t, fuse_softmax_backward, packed_input, for_vector_kernel):
        self.gen_input(scalar_t, for_vector_kernel)
        my_loss = TransducerLoss(  fuse_softmax_backward=fuse_softmax_backward, 
                                    packed_input=packed_input) 
        if not packed_input:
            loss_tst = my_loss( x=self.x_tst,
                                label=self.y, 
                                f_len=self.f_len, 
                                y_len=self.y_len, 
                                blank_idx=self.blank_idx)
            loss_tst.mean().backward() 
            grad_tst = self.x_tst.grad
        else:
            loss_tst = my_loss( x=self.x_tst_packed,
                                label=self.y, 
                                f_len=self.f_len, 
                                y_len=self.y_len, 
                                blank_idx=self.blank_idx,
                                batch_offset=self.batch_offset, 
                                max_f_len=max(self.f_len))
            loss_tst.mean().backward()
            grad_tst_packed = self.x_tst_packed.grad
            grad_tst = self._unpack(grad_tst_packed)
        
        return loss_tst, grad_tst

    def test_transducer_loss_fp32(self):
        loss_tst, grad_tst = self.run_transducer_loss(  scalar_t=torch.float32,
                                                        fuse_softmax_backward=False,
                                                        packed_input=False,
                                                        for_vector_kernel=False)
        self.assertTrue(torch.allclose(self.loss_ref, loss_tst, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(self.grad_ref, grad_tst, atol=1e-5, rtol=1e-5))

    def test_transducer_loss_fp16(self):
        loss_tst, grad_tst = self.run_transducer_loss(  scalar_t=torch.float16,
                                                        fuse_softmax_backward=False,
                                                        packed_input=False,
                                                        for_vector_kernel=False)
        self.assertTrue(torch.allclose(self.loss_ref, loss_tst, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(self.grad_ref, grad_tst, atol=1e-4, rtol=1e-3))

    def test_transducer_loss_fp16_backward_fusion(self):
        loss_tst, grad_tst = self.run_transducer_loss(  scalar_t=torch.float16,
                                                        fuse_softmax_backward=True,
                                                        packed_input=False,
                                                        for_vector_kernel=False)
        self.assertTrue(torch.allclose(self.loss_ref, loss_tst, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(self.grad_ref, grad_tst, atol=1e-4, rtol=1e-3))

    def test_transducer_loss_fp16_backward_fusion_packed(self):
        loss_tst, grad_tst = self.run_transducer_loss(  scalar_t=torch.float16,
                                                        fuse_softmax_backward=True,
                                                        packed_input=True,
                                                        for_vector_kernel=False)
        self.assertTrue(torch.allclose(self.loss_ref, loss_tst, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(self.grad_ref, grad_tst, atol=1e-4, rtol=1e-3))

    def test_transducer_loss_fp16_backward_fusion_packed_vec(self):
        loss_tst, grad_tst = self.run_transducer_loss(  scalar_t=torch.float16,
                                                        fuse_softmax_backward=True,
                                                        packed_input=True,
                                                        for_vector_kernel=True)
        self.assertTrue(torch.allclose(self.loss_ref, loss_tst, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(self.grad_ref, grad_tst, atol=1e-4, rtol=1e-3))



if __name__ == '__main__':
    unittest.main()