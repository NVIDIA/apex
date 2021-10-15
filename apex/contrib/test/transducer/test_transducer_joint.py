import torch
import unittest
from apex.contrib.transducer import TransducerJoint
import transducer_ref

class TransducerJointTest(unittest.TestCase):
    def setUp(self, seed=1234):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def gen_input(self, for_vector_kernel):
        self.B = 4
        T_min = 51
        T_max = 101
        U_min = 12
        U_max = 25
        if for_vector_kernel:
            H = 512
        else:
            H = 509
        dtype = torch.float16
        device = "cuda"

        self.f_tst = torch.randn((self.B, T_max, H), dtype=dtype, requires_grad=True, device=device)
        self.g_tst = torch.randn((self.B, U_max, H), dtype=dtype, requires_grad=True, device=device)
        self.h_grad = torch.randn(self.B, T_max, U_max, H, dtype=dtype, device=device)
        self.f_len = torch.randint(T_min, T_max+1, (self.B,), dtype=torch.int, device=device) 
        self.g_len = torch.randint(U_min, U_max+1, (self.B,), dtype=torch.int, device=device)
        self.f_len[torch.randint(0, self.B, (1,)).item()] = T_max
        self.g_len[torch.randint(0, self.B, (1,)).item()] = U_max
        self.dropout_prob = 0.5

        # Make sure gradients from out-of-bound locations are zero. This should be guaranteed by 
        # the loss function
        for b in range(self.B):
            self.h_grad[b, self.f_len[b]:, :, :] = 0
            self.h_grad[b, :, self.g_len[b]:, :] = 0
        self.h_grad_packed = self._pack(self.h_grad, self.f_len, self.g_len)
        

    def _pack(self, x, f_len, g_len):
        B = x.size(0)
        list_x = []
        for b in range(B):
            list_x_row = [x[b, t, :g_len[b]] for t in range(f_len[b])]
            x_row = torch.cat(list_x_row)
            list_x.append(x_row)
        x_packed = torch.cat(list_x).data.clone()
        x_packed.requires_grad = True
        batch_offset = torch.cumsum(f_len * g_len, dim=0)
        return x_packed

    def _unpack(self, x, f_len, g_len):
        batch_offset = torch.cumsum(f_len * g_len, dim=0)
        x_unpacked = torch.zeros_like(self.h_grad, dtype=torch.uint8)
        B = self.h_grad.size(0)
        H = self.h_grad.size(-1)
        for b in range(B):
            my_batch_offset = 0 if b == 0 else batch_offset[b-1]
            my_f_len = f_len[b]
            my_g_len = g_len[b]
            for t in range(my_f_len):
                x_unpacked[b, t, :my_g_len] = x[my_batch_offset + t*my_g_len : 
                                                my_batch_offset + t*my_g_len + my_g_len]
        return x_unpacked
        
    def run_transducer_joint(self, for_vector_kernel, pack_output, relu, dropout):
        self.gen_input(for_vector_kernel=for_vector_kernel)
        # Generate reference
        f_ref = self.f_tst.data.clone()
        g_ref = self.g_tst.data.clone()
        f_ref.requires_grad = True
        g_ref.requires_grad = True
        
        my_joint = TransducerJoint(pack_output=pack_output, relu=relu, dropout=dropout, 
                                    dropout_prob=self.dropout_prob, probe_mask=True)
        if not pack_output:
            h_tst = my_joint(   f=self.f_tst, 
                                g=self.g_tst, 
                                f_len=self.f_len, 
                                g_len=self.g_len)
            h_tst.backward(self.h_grad)
            if dropout:
                mask = my_joint.mask_probe[0]
        else:
            batch_offset = torch.cumsum(self.f_len * self.g_len, dim=0)
            h_tst = my_joint(   f=self.f_tst, 
                                g=self.g_tst, 
                                f_len=self.f_len, 
                                g_len=self.g_len, 
                                batch_offset=batch_offset, 
                                packed_batch=batch_offset[-1])
            h_tst.backward(self.h_grad_packed)
            if dropout:
                mask_packed = my_joint.mask_probe[0]
                mask = self._unpack(mask_packed, self.f_len, self.g_len)

        # reference
        h_ref, f_grad_ref, g_grad_ref \
            = transducer_ref.transducer_joint_reference(f=f_ref, 
                                                        g=g_ref, 
                                                        h_grad=self.h_grad, 
                                                        f_len=self.f_len, 
                                                        g_len=self.g_len, 
                                                        pack_output=pack_output,
                                                        relu=relu,
                                                        dropout=dropout,
                                                        dropout_prob=self.dropout_prob,
                                                        mask=mask if dropout else None)
        
        f_grad_tst = self.f_tst.grad
        g_grad_tst = self.g_tst.grad
        
        self.assertTrue(torch.allclose(h_ref, h_tst, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(f_grad_ref, f_grad_tst, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(g_grad_ref, g_grad_tst, atol=1e-4, rtol=1e-4))

    def test_transducer_joint(self):
        self.run_transducer_joint(for_vector_kernel=True, pack_output=True, relu=False, dropout=False)

    def test_transducer_joint_vec(self):
        self.run_transducer_joint(for_vector_kernel=True, pack_output=False, relu=False, dropout=False)

    def test_transducer_joint_pack(self):
        self.run_transducer_joint(for_vector_kernel=False, pack_output=True, relu=False, dropout=False)

    def test_transducer_joint_vec_pack(self):
        self.run_transducer_joint(for_vector_kernel=True, pack_output=True, relu=False, dropout=False)

    def test_transducer_joint_relu(self):
        self.run_transducer_joint(for_vector_kernel=True, pack_output=True, relu=True, dropout=False)

    def test_transducer_joint_vec_relu(self):
        self.run_transducer_joint(for_vector_kernel=True, pack_output=False, relu=True, dropout=False)

    def test_transducer_joint_pack_relu(self):
        self.run_transducer_joint(for_vector_kernel=False, pack_output=True, relu=True, dropout=False)

    def test_transducer_joint_vec_pack_relu(self):
        self.run_transducer_joint(for_vector_kernel=True, pack_output=True, relu=True, dropout=False)

    def test_transducer_joint_relu_dropout(self):
        self.run_transducer_joint(for_vector_kernel=True, pack_output=True, relu=True, dropout=True)

    def test_transducer_joint_vec_relu_dropout(self):
        self.run_transducer_joint(for_vector_kernel=True, pack_output=False, relu=True, dropout=True)

    def test_transducer_joint_pack_relu_dropout(self):
        self.run_transducer_joint(for_vector_kernel=False, pack_output=True, relu=True, dropout=True)

    def test_transducer_joint_vec_pack_relu_dropout(self):
        self.run_transducer_joint(for_vector_kernel=True, pack_output=True, relu=True, dropout=True)



if __name__ == '__main__':
    unittest.main()