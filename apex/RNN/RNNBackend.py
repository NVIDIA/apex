import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F

import math


def is_iterable(maybe_iterable):
    return isinstance(maybe_iterable, list) or isinstance(maybe_iterable, tuple)


def flatten_list(tens_list):
    """
    flatten_list
    """
    if not is_iterable(tens_list):
        return tens_list
    
    return torch.cat(tens_list, dim=0).view(len(tens_list), *tens_list[0].size() )

    
#These modules always assumes batch_first
class bidirectionalRNN(nn.Module):
    """
    bidirectionalRNN
    """
    def __init__(self, inputRNN, num_layers=1, dropout = 0):
        super(bidirectionalRNN, self).__init__()
        self.dropout = dropout
        self.fwd = stackedRNN(inputRNN, num_layers=num_layers, dropout = dropout)
        self.bckwrd = stackedRNN(inputRNN.new_like(), num_layers=num_layers, dropout = dropout)
        self.rnns = nn.ModuleList([self.fwd, self.bckwrd])
        
    #collect hidden option will return all hidden/cell states from entire RNN
    def forward(self, input, collect_hidden=False):
        """
        forward()
        """
        seq_len = input.size(0)
        bsz = input.size(1)

        fwd_out, fwd_hiddens = list(self.fwd(input, collect_hidden = collect_hidden))
        bckwrd_out, bckwrd_hiddens = list(self.bckwrd(input, reverse=True, collect_hidden = collect_hidden))
        
        output = torch.cat( [fwd_out, bckwrd_out], -1 )
        hiddens = tuple( torch.cat(hidden, -1) for hidden in zip( fwd_hiddens, bckwrd_hiddens) )

        return output, hiddens

    def reset_parameters(self):
        """
        reset_parameters()
        """
        for rnn in self.rnns:
            rnn.reset_parameters()
        
    def init_hidden(self, bsz):
        """
        init_hidden()
        """
        for rnn in self.rnns:
            rnn.init_hidden(bsz)

    def detach_hidden(self):
        """
        detach_hidden()
        """
        for rnn in self.rnns:
            rnn.detachHidden()
        
    def reset_hidden(self, bsz):
        """
        reset_hidden()
        """
        for rnn in self.rnns:
            rnn.reset_hidden(bsz)

    def init_inference(self, bsz):    
        """
        init_inference()
        """
        for rnn in self.rnns:
            rnn.init_inference(bsz)

   
#assumes hidden_state[0] of inputRNN is output hidden state
#constructor either takes an RNNCell or list of RNN layers
class stackedRNN(nn.Module):        
    """
    stackedRNN
    """
    def __init__(self, inputRNN, num_layers=1, dropout=0):
        super(stackedRNN, self).__init__()
        
        self.dropout = dropout
        
        if isinstance(inputRNN, RNNCell):
            self.rnns = [inputRNN]
            for i in range(num_layers-1):
                self.rnns.append(inputRNN.new_like(inputRNN.output_size))
        elif isinstance(inputRNN, list):
            assert len(inputRNN) == num_layers, "RNN list length must be equal to num_layers"
            self.rnns=inputRNN
        else:
            raise RuntimeError()
        
        self.nLayers = len(self.rnns)
        
        self.rnns = nn.ModuleList(self.rnns)


    '''
    Returns output as hidden_state[0] Tensor([sequence steps][batch size][features])
    If collect hidden will also return Tuple(
        [n_hidden_states][sequence steps] Tensor([layer][batch size][features])
    )
    If not collect hidden will also return Tuple(
        [n_hidden_states] Tensor([layer][batch size][features])
    '''
    def forward(self, input, collect_hidden=False, reverse=False):
        """
        forward()
        """
        seq_len = input.size(0)
        bsz = input.size(1)
        inp_iter = reversed(range(seq_len)) if reverse else range(seq_len)

        hidden_states = [[] for i in range(self.nLayers)]
        outputs = []

        for seq in inp_iter:
            for layer in range(self.nLayers):

                if layer == 0:
                    prev_out = input[seq]
                    
                outs = self.rnns[layer](prev_out)

                if collect_hidden:
                    hidden_states[layer].append(outs)
                elif seq == seq_len-1:
                    hidden_states[layer].append(outs)
                    
                prev_out = outs[0]

            outputs.append(prev_out)

        if reverse:
            outputs = list(reversed(outputs))
        '''
        At this point outputs is in format:
        list( [seq_length] x Tensor([bsz][features]) )
        need to convert it to:
        list( Tensor([seq_length][bsz][features]) )
        '''
        output = flatten_list(outputs)

        '''
        hidden_states at this point is in format:
        list( [layer][seq_length][hidden_states] x Tensor([bsz][features]) )
        need to convert it to:
          For not collect hidden:
            list( [hidden_states] x Tensor([layer][bsz][features]) )
          For collect hidden:
            list( [hidden_states][seq_length] x Tensor([layer][bsz][features]) )
        '''
        if not collect_hidden:
            seq_len = 1
        n_hid = self.rnns[0].n_hidden_states
        new_hidden = [ [ [ None for k in range(self.nLayers)] for j in range(seq_len) ] for i in range(n_hid) ]


        for i in range(n_hid):
            for j in range(seq_len):
                for k in range(self.nLayers):
                    new_hidden[i][j][k] = hidden_states[k][j][i]

        hidden_states = new_hidden
        #Now in format list( [hidden_states][seq_length][layer] x Tensor([bsz][features]) )
        #Reverse seq_length if reverse
        if reverse:
            hidden_states = list( list(reversed(list(entry))) for entry in hidden_states)

        #flatten layer dimension into tensor
        hiddens = list( list(
            flatten_list(seq) for seq in hidden )
                        for hidden in hidden_states )
        
        #Now in format list( [hidden_states][seq_length] x Tensor([layer][bsz][features]) )
        #Remove seq_length dimension if not collect_hidden
        if not collect_hidden:
            hidden_states = list( entry[0] for entry in hidden_states)
        return output, hidden_states
    
    def reset_parameters(self):
        """
        reset_parameters()
        """
        for rnn in self.rnns:
            rnn.reset_parameters()
        
    def init_hidden(self, bsz):
        """
        init_hidden()
        """
        for rnn in self.rnns:
            rnn.init_hidden(bsz)

    def detach_hidden(self):
        """
        detach_hidden()
        """
        for rnn in self.rnns:
            rnn.detach_hidden()
        
    def reset_hidden(self, bsz):
        """
        reset_hidden()
        """
        for rnn in self.rnns:
            rnn.reset_hidden(bsz)

    def init_inference(self, bsz):    
        """ 
        init_inference()
        """
        for rnn in self.rnns:
            rnn.init_inference(bsz)

class RNNCell(nn.Module):
    """ 
    RNNCell 
    gate_multiplier is related to the architecture you're working with
    For LSTM-like it will be 4 and GRU-like will be 3.
    Always assumes input is NOT batch_first.
    Output size that's not hidden size will use output projection
    Hidden_states is number of hidden states that are needed for cell
    if one will go directly to cell as tensor, if more will go as list
    """
    def __init__(self, gate_multiplier, input_size, hidden_size, cell, n_hidden_states = 2, bias = False, output_size = None):
        super(RNNCell, self).__init__()

        self.gate_multiplier = gate_multiplier
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.bias = bias
        self.output_size = output_size
        if output_size is None:
            self.output_size = hidden_size

        self.gate_size = gate_multiplier * self.hidden_size
        self.n_hidden_states = n_hidden_states

        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.output_size))

        #Check if there's recurrent projection
        if(self.output_size != self.hidden_size):
            self.w_ho = nn.Parameter(torch.Tensor(self.output_size, self.hidden_size))

        self.b_ih = self.b_hh = None
        if self.bias:
            self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
            self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
            
        #hidden states for forward
        self.hidden = [ None for states in range(self.n_hidden_states)]

        self.reset_parameters()

    def new_like(self, new_input_size=None):
        """
        new_like()
        """
        if new_input_size is None:
            new_input_size = self.input_size
            
        return type(self)(self.gate_multiplier,
                       new_input_size,
                       self.hidden_size,
                       self.cell,
                       self.n_hidden_states,
                       self.bias,
                       self.output_size)

    
    #Use xavier where we can (weights), otherwise use uniform (bias)
    def reset_parameters(self, gain=1):
        """
        reset_parameters()
        """
        stdev = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            param.data.uniform_(-stdev, stdev)
    '''
    Xavier reset:
    def reset_parameters(self, gain=1):
        stdv = 1.0 / math.sqrt(self.gate_size)

        for param in self.parameters():
            if (param.dim() > 1):
                torch.nn.init.xavier_normal(param, gain)
            else:
                param.data.uniform_(-stdv, stdv)
    '''
    def init_hidden(self, bsz):
        """
        init_hidden()
        """
        for param in self.parameters():
            if param is not None:
                a_param = param
                break

        for i, _ in enumerate(self.hidden):
            if(self.hidden[i] is None or self.hidden[i].data.size()[0] != bsz):

                if i==0:
                    hidden_size = self.output_size
                else:
                    hidden_size = self.hidden_size

                tens = a_param.data.new(bsz, hidden_size).zero_()
                self.hidden[i] = Variable(tens, requires_grad=False)
            
        
    def reset_hidden(self, bsz):
        """
        reset_hidden()
        """
        for i, _ in enumerate(self.hidden):
            self.hidden[i] = None
        self.init_hidden(bsz)

    def detach_hidden(self):
        """
        detach_hidden()
        """
        for i, _ in enumerate(self.hidden):
            if self.hidden[i] is None:
                raise RuntimeError("Must initialize hidden state before you can detach it")
        for i, _ in enumerate(self.hidden):
            self.hidden[i] = self.hidden[i].detach()
        
    def forward(self, input):
        """
        forward()
        if not inited or bsz has changed this will create hidden states
        """
        self.init_hidden(input.size()[0])

        hidden_state = self.hidden[0] if self.n_hidden_states == 1 else self.hidden
        self.hidden = self.cell(input, hidden_state, self.w_ih, self.w_hh, b_ih=self.b_ih, b_hh=self.b_hh)
        if(self.n_hidden_states > 1):
            self.hidden = list(self.hidden)
        else:
            self.hidden=[self.hidden]

        if self.output_size != self.hidden_size:
            self.hidden[0] = F.linear(self.hidden[0], self.w_ho)

        return tuple(self.hidden)
