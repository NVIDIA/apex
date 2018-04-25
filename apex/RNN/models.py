import torch

from torch.nn._functions.rnn import LSTMCell, RNNReLUCell, RNNTanhCell, GRUCell

from .RNNBackend import bidirectionalRNN, stackedRNN, RNNCell
from .cells import mLSTMRNNCell, mLSTMCell

def toRNNBackend(inputRNN, num_layers, bidirectional=False, dropout = 0):
    """
    :class:`toRNNBackend`
    """

    if bidirectional:
        return bidirectionalRNN(inputRNN, num_layers, dropout = dropout)
    else:
        return stackedRNN(inputRNN, num_layers, dropout = dropout)


def LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, output_size = None):
    """
    :class:`LSTM`
    """
    inputRNN = RNNCell(4, input_size, hidden_size, LSTMCell, 2, bias, output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)

def GRU(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, output_size = None):
    """
    :class:`GRU`
    """
    inputRNN = RNNCell(3, input_size, hidden_size, GRUCell, 1, bias, output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)

def ReLU(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, output_size = None):
    """
    :class:`ReLU`
    """
    inputRNN = RNNCell(1, input_size, hidden_size, RNNReLUCell, 1, bias, output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)

def Tanh(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, output_size = None):
    """
    :class:`Tanh`
    """
    inputRNN = RNNCell(1, input_size, hidden_size, RNNTanhCell, 1, bias, output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)
        
def mLSTM(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, output_size = None):
    """
    :class:`mLSTM`
    """
    inputRNN = mLSTMRNNCell(input_size, hidden_size, bias=bias, output_size=output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)


