import torch
import transducer_loss_cuda
import transducer_joint_cuda

class TransducerJoint(torch.nn.Module):
    """Transducer joint
    Detail of this loss function can be found in: Sequence Transduction with Recurrent Neural 
    Networks

    Arguments:
        pack_output (bool, optional): whether to pack the output in a compact form with don't-care 
        data being removed. (default: False)
        opt (int, optional): pick the optimization level in [0, 1]. opt=1 picks a tiled algorithm. 
            (default: 1)
        fwd_tile_size (int, optional): tile size used in forward operation. This argument will be 
        ignored if opt != 1. (default: 4) 
    """

    def __init__(self, pack_output=False, opt=1, fwd_tile_size=4):
        super(TransducerJoint, self).__init__() 
        self.pack_output = pack_output
        self.opt = opt
        self.fwd_tile_size = fwd_tile_size
        self.dummy_batch_offset = torch.empty(0)


    def forward(self, f, g, f_len, g_len, batch_offset=None, packed_batch=0):
        """Forward operation of transducer joint

        Arguments:
            f (tensor): transcription vector from encode block of shape (B, T, H).
            g (tensor): prediction vector form predict block of shape (B, U, H).
            f_len (tensor): length of transcription vector for each batch.
            g_len (tensor): length of prediction vector minus 1 for each batch.
            batch_offset (tensor, optional): tensor containing the offset of each batch
                in the results. For example, batch offset can be obtained from: 
                batch_offset = torch.cumsum(f_len*g_len, dim=0)
                This argument is required if pack_output == True, and is ignored if 
                pack_output == False. (default: None)
            packed_batch (int, optional): the batch size after packing. This argument is 
                ignored if pack_output == False. (default: 0)
        """
        my_batch_offset = batch_offset if self.pack_output else self.dummy_batch_offset
        if self.pack_output and (batch_offset is None or packed_batch == 0):
            raise Exception("Please specify batch_offset and packed_batch when packing is enabled")
        return TransducerJointFunc.apply(f, g, f_len, g_len, self.pack_output, my_batch_offset, 
                                            packed_batch, self.opt, self.fwd_tile_size)


class TransducerLoss(torch.nn.Module):
    """Transducer loss
    Detail of this loss function can be found in: Sequence Transduction with Recurrent Neural 
    Networks

    Arguments:
        fuse_softmax_backward (bool, optional) whether to fuse the backward of transducer loss with
            softmax. (default: True)
        opt (int, optional): pick the optimization level in [0, 1]. opt=1 picks a more optimized 
            algorithm. In some cases, opt=1 might fall back to opt=0. (default: 1)
        packed_input (bool, optional): whether to pack the output in a compact form with don't-care 
        data being removed. (default: False)
    """
    def __init__(self, fuse_softmax_backward=True, opt=1, packed_input=False):
        super(TransducerLoss, self).__init__() 
        self.fuse_softmax_backward = fuse_softmax_backward
        self.opt = opt
        self.packed_input = packed_input
        self.dummy_batch_offset = torch.empty(0)


    def forward(self, x, label, f_len, y_len, blank_idx, batch_offset=None, max_f_len=None, 
                debug_list=None):
        """Forward operation of transducer joint

        Arguments:
            x (tensor): input tensor to the loss function with a shape of (B, T, U, H).
            label (tensor): labels for the input data.
            f_len (tensor): lengths of the inputs in the time dimension for each batch.
            y_len (tensor): lengths of the labels for each batch.
            blank_idx (int): index for the null symbol.
            batch_offset (tensor, optional): tensor containing the offset of each batch
                in the input. For example, batch offset can be obtained from: 
                batch_offset = torch.cumsum(f_len*(y_len+1), dim=0)
                This argument is required if packed_input == True, and is ignored if 
                packed_input == False. (default: None)
            max_f_len (int, optional): maximum length of the input in the time dimension.
                For example, it can be obtained as 
                max_f_len = max(f_len)
                This argument is required if packed_input == True, and is ignored if 
                packed_input == False. (default: None)
                (default: None)
            debug_list (list, optional): when an empty list is supplied, Alpha and Beta generated 
                in the forward operation will be attached to this list for debug purpose. 
                (default: None)
        """
        if self.packed_input:
            if batch_offset is None or max_f_len is None:
                raise Exception("Please specify batch_offset and max_f_len when packing is \
                                    enabled") 
            my_batch_offset = batch_offset
            my_max_f_len = max_f_len
        else:
            my_batch_offset = self.dummy_batch_offset
            my_max_f_len = x.size(1)
        return TransducerLossFunc.apply(x, label, f_len, y_len, my_batch_offset, my_max_f_len, 
                                            blank_idx, self.fuse_softmax_backward, debug_list, 
                                            self.opt, self.packed_input)

class TransducerLossFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, label, f_len, y_len, batch_offset, max_f_len, blank_idx, 
                fuse_softmax_backward, debug_list, opt, packed_input):
        if fuse_softmax_backward == False:
            with torch.enable_grad():
                x = torch.nn.functional.log_softmax(x, dim=-1)
        else:
            x = torch.nn.functional.log_softmax(x, dim=-1)
        alpha, beta, loss = transducer_loss_cuda.forward(   x, label, f_len, y_len, batch_offset, 
                                                            max_f_len, blank_idx, opt, packed_input)
        if debug_list == []:
            debug_list += [alpha, beta]
        ctx.save_for_backward(x, alpha, beta, f_len, y_len, label, batch_offset)
        ctx.blank_idx = blank_idx
        ctx.fuse_softmax_backward = fuse_softmax_backward
        ctx.opt = opt
        ctx.packed_input = packed_input
        ctx.max_f_len = max_f_len
        return loss

    @staticmethod
    def backward(ctx, loss_grad):
        x, alpha, beta, f_len, y_len, label, batch_offset = ctx.saved_tensors
        x_grad = transducer_loss_cuda.backward( x, loss_grad, alpha, beta, f_len, y_len, label, 
                                                batch_offset, ctx.max_f_len, ctx.blank_idx, ctx.opt, 
                                                ctx.fuse_softmax_backward, ctx.packed_input)
        if ctx.fuse_softmax_backward == False:
            x_grad = x.backward(x_grad)
        return x_grad, None, None, None, None, None, None, None, None, None, None

class TransducerJointFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, g, f_len, g_len, pack_output, batch_offset, packed_batch, opt, 
                fwd_tile_size):
        h = transducer_joint_cuda.forward(f, g, f_len, g_len, batch_offset, packed_batch, opt, 
                                            pack_output, fwd_tile_size)
        ctx.save_for_backward(f_len, g_len, batch_offset)
        ctx.pack_output = pack_output
        ctx.max_f_len = f.size(1)
        ctx.max_g_len = g.size(1)
        return h

    @staticmethod
    def backward(ctx, loss_grad):
        f_len, g_len, batch_offset = ctx.saved_tensors
        f_grad, g_grad = transducer_joint_cuda.backward(loss_grad, f_len, g_len, batch_offset, 
                                                        ctx.max_f_len, ctx.max_g_len, 
                                                        ctx.pack_output)

        return f_grad, g_grad, None, None, None, None, None, None, None, None, None, None


