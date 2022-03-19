import torch
import numpy as np
import pdb

def transducer_loss_reference(x, label, f_len, y_len, blank_idx, loss_grad):
    def log_sum_exp(a, b):
        if (a >= b):
            return a + torch.log(1 + torch.exp(b-a))
        else:
            return b + torch.log(1 + torch.exp(a-b))

    def forward_alpha(x, label, f_len, y_len, blank_idx):
        B, T, U, V = x.size()
        acc_t = torch.float32 if x.dtype in [torch.float16, torch.float32] else x.dtype
        alpha = torch.zeros((B, T, U), dtype=acc_t, device=x.device)
        for b in range(B):
            alpha[b, 0, 0] = 0
            for t in range(1, f_len[b]):
                alpha[b, t, 0] = alpha[b, t-1, 0] + x[b, t-1, 0, blank_idx]
            for u in range(1, y_len[b]+1):
                alpha[b, 0, u] = alpha[b, 0, u-1] + x[b, 0, u-1, label[b, u-1]]
            for t in range(1, f_len[b]):
                for u in range(1, y_len[b]+1):
                    curr_ = alpha[b, t-1, u] + x[b, t-1, u, blank_idx]
                    next_ = alpha[b, t, u-1] + x[b, t, u-1, label[b, u-1]]
                    alpha[b, t, u] = log_sum_exp(curr_, next_) 
        return alpha

    def forward_beta(x, label, f_len, y_len, blank_idx):
        B, T, U, V = x.shape
        acc_t = torch.float32 if x.dtype in [torch.float16, torch.float32] else x.dtype
        beta = torch.zeros((B, T, U), dtype=acc_t, device=x.device)
        for b in range(B):
            beta[b, f_len[b]-1, y_len[b]] = x[b, f_len[b]-1, y_len[b], blank_idx]
            for t in range(f_len[b]-2, -1, -1):
                beta[b, t, y_len[b]] = beta[b, t+1, y_len[b]] + x[b, t, y_len[b], blank_idx] 
            for u in range(y_len[b]-1, -1, -1):
                beta[b, f_len[b]-1, u] = beta[b, f_len[b]-1, u+1] + x[b, f_len[b]-1, u, label[b, u]]
            for t in range(f_len[b]-2, -1, -1):
                for u in range(y_len[b]-1, -1, -1):
                    curr_ = beta[b, t+1, u] + x[b, t, u, blank_idx] 
                    next_ = beta[b, t, u+1] + x[b, t, u, label[b, u]]
                    beta[b, t, u] = log_sum_exp(curr_, next_) 
        return beta

    def backward(x, label, f_len, y_len, alpha, beta, loss_grad, blank_idx):
        grad = torch.zeros_like(x)
        B, T, U, V = x.size()
        for b in range(B):
            common_factor = torch.log(loss_grad[b]) + alpha - beta[b, 0, 0]
            # next
            for u in range(y_len[b]):
                grad[b, :f_len[b], u, label[b, u]] = -torch.exp(common_factor[b, :f_len[b], u] 
                                                        + beta[b, :f_len[b], u+1] 
                                                        + x[b, :f_len[b], u, label[b, u]])

            # current
            grad[b, :f_len[b]-1, :y_len[b]+1, blank_idx] \
                = -torch.exp(common_factor[b, :f_len[b]-1, :y_len[b]+1] 
                    + beta[b, 1:f_len[b], :y_len[b]+1] 
                    + x[b, :f_len[b]-1, :y_len[b]+1, blank_idx])

            grad[b, f_len[b]-1, y_len[b], blank_idx] = -torch.exp(common_factor[b, f_len[b]-1, y_len[b]]
                                                         + x[b, f_len[b]-1, y_len[b], blank_idx])
     
        return grad

    x_log = torch.nn.functional.log_softmax(x, dim=-1)
    alpha = forward_alpha(x_log, label, f_len, y_len, blank_idx)
    beta = forward_beta(x_log, label, f_len, y_len, blank_idx)
    grad = backward(x_log, label, f_len, y_len, alpha, beta, 
                        loss_grad, blank_idx)
    x_log.backward(grad)
    loss = -beta[:, 0, 0]
    loss = loss.to(x.dtype)
    return alpha, beta, x.grad, loss


def transducer_joint_reference(f, g, h_grad, f_len, g_len, pack_output, relu, dropout, 
                                dropout_prob=0, mask=None):
    if dropout and mask == None:
        raise NotImplementedError("mask needs to supplied to test dropout.")
    B, T, H = f.size()
    U = g.size(1)
    f_expand = f.unsqueeze(dim=2)
    g_expand = g.unsqueeze(dim=1)
    h = f_expand + g_expand
    if relu:
        h = torch.nn.functional.relu(h)
    if dropout:
        h *= mask
        scale = 1/(1-dropout_prob)
        h *= scale
    h.backward(h_grad)

    if pack_output == False:
        # intentionally set don't-care region to -1 to test if transducer joint
        # write these regions to avoid NaN and inf
        for b in range(B):
            h[b, f_len[b]:] = -1
            h[b, :, g_len[b]:] = -1

        return h, f.grad, g.grad 

    # packing
    list_to_pack = []
    for b in range(B):
        list_to_pack.append(h[b, :f_len[b], :g_len[b], :].reshape(-1, H))
    h_packed = torch.cat(list_to_pack)
    return h_packed, f.grad, g.grad


