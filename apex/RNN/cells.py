import torch
import torch.nn as nn
import torch.nn.functional as F

from .RNNBackend import RNNCell

from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

import math 


class mLSTMRNNCell(RNNCell):
    """
    mLSTMRNNCell
    """

    def __init__(self, input_size, hidden_size, bias = False, output_size = None):
        gate_multiplier = 4
        super(mLSTMRNNCell, self).__init__(gate_multiplier, input_size, hidden_size, mLSTMCell, n_hidden_states = 2, bias = bias, output_size = output_size)

        self.w_mih = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.w_mhh = nn.Parameter(torch.empty(self.output_size, self.output_size))

        self.reset_parameters()

    def forward(self, input):
        """
        mLSTMRNNCell.forward()
        """
        #if not inited or bsz has changed this will create hidden states
        self.init_hidden(input.size()[0])

        hidden_state = self.hidden[0] if self.n_hidden_states == 1 else self.hidden

        self.hidden = list(
                           self.cell(input, hidden_state, self.w_ih, self.w_hh, self.w_mih, self.w_mhh,
                           b_ih=self.b_ih, b_hh=self.b_hh)
        )
        
        if self.output_size != self.hidden_size:
            self.hidden[0] = F.linear(self.hidden[0], self.w_ho)
        return tuple(self.hidden)


    def new_like(self, new_input_size=None):
        if new_input_size is None:
            new_input_size = self.input_size
        
        return type(self)(
            new_input_size,
            self.hidden_size,
            self.bias,
            self.output_size)

def mLSTMCell(input, hidden, w_ih, w_hh, w_mih, w_mhh, b_ih=None, b_hh=None):
    """
    mLSTMCell
    """

    if input.is_cuda:
        igates = F.linear(input, w_ih)
        m = F.linear(input, w_mih) * F.linear(hidden[0], w_mhh)
        hgates = F.linear(m, w_hh)

        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    
    m = F.linear(input, w_mih) * F.linear(hidden[0], w_mhh)
    gates = F.linear(input, w_ih, b_ih) + F.linear(m, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)
    
    return hy, cy
                                                                            
