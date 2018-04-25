import torch
import torch.nn as nn
from torch.autograd import Variable
import apex
from apex.RNN.models import bidirectionalRNN, stackedRNN, RNNCell
from torch.nn._functions.rnn import LSTMCell
import itertools


torch.backends.cudnn.enabled=False

batch_first = False #not implemented yet
dropout = 0.0 #How to validate?
bidirectional = False #True works, but differs in definition to PyTorch

rnn_types = ['LSTM', 'GRU', 'ReLU', 'Tanh']
sizes = [8,4,2]

seq_sizes = sizes
hidden_sizes = sizes
inp_sizes = sizes
batch_sizes = sizes
num_layerss = sizes

biases = [True]

def copy_param_set(pyt_rnn, my_rnn, layer=0, reverse=False):
    my_params = None

    rnn = None
    if isinstance(my_rnn, bidirectionalRNN):
        rnn = my_rnn.fwd.rnns[layer] if not reverse else my_rnn.bckwrd.rnns[layer]
    elif isinstance(my_rnn, stackedRNN):
        rnn = my_rnn.rnns[layer]
    else:
        raise RuntimeError()

    param_names = ['w_ih', 'w_hh', 'b_ih', 'b_hh']

    if not hasattr(rnn, 'b_hh'):
        param_names = param_names[:2]
    my_params = [getattr(rnn, param_name) for param_name in param_names]
        
    pyt_params = None
    param_names = ['weight_ih_', 'weight_hh_', 'bias_ih_', 'bias_hh_']
    reverse_str = '_reverse' if reverse else ''

    if not hasattr(pyt_rnn, 'bias_hh_l0'):
        param_names=param_names[:2]
    pyt_params =[getattr(pyt_rnn, param_name + 'l' + str(layer) + reverse_str )
                 for param_name in param_names ]
    for pyt_param, my_param in zip(pyt_params, my_params):
        pyt_param.data.copy_(my_param.data)

def copy_all_params(pyt_rnn, my_rnn):
    for layer in range(num_layers):
        copy_param_set(pyt_rnn, my_rnn, layer)
        if bidirectional:
            copy_param_set(pyt_rnn, my_rnn, layer, bidirectional)


def compare_variables(v1, v2, msg, params):
    diff = float((v1.data-v2.data).abs().max())
    if diff > 1e-5:
        print("Error of ", diff, " found for ", msg, " for case: ", str(params))
    
def compare_tuple_variables(t1, t2, msg, params):
    for var1, var2 in zip(t1, t2):
        compare_variables(var1, var2, msg, params)

def maybe_compare(v1, v2, msg, params):
    if isinstance(v1, Variable) and isinstance(v2, Variable):
        compare_variables(v1, v2, msg, params)
    else:
        compare_tuple_variables(v1, v2, msg, params)

product = list(itertools.product(rnn_types, seq_sizes, hidden_sizes, inp_sizes, batch_sizes, num_layerss, biases))

for test_case in product:
    rnn_type, seq_size, hidden_size, inp_size, batch_size, num_layers, bias = test_case

    inp = torch.cuda.FloatTensor(seq_size, batch_size, inp_size).uniform_()

    if rnn_type == 'ReLU' or rnn_type == 'Tanh':
        pytorch_rnn = nn.RNN(inp_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, nonlinearity=rnn_type.lower()).cuda()
    else:
        pytorch_rnn =     getattr(nn, rnn_type)(inp_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional).cuda()
    my_rnn = getattr(apex.RNN.models, rnn_type)(inp_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional).cuda()
    
    copy_all_params(pytorch_rnn, my_rnn)

    pyt_inp = Variable(inp, requires_grad=True)
    my_inp  = Variable(inp, requires_grad=True)

    my_out, my_hiddens =  my_rnn(my_inp)
    pyt_out, pyt_hiddens = pytorch_rnn(pyt_inp)

    pyt_out.sum().backward()
    my_out.sum().backward()


    maybe_compare(pyt_out, my_out, "out", test_case)

    #If there's only one hidden state PyTorch doesn't return it in a tuple,
    #apex does, so we wrap PyTorch's returned hidden state in a tuple.
    if not isinstance(pyt_hiddens, tuple):
        pyt_hiddens = (pyt_hiddens,)

    try:
        for i, (pyt_hid, my_hid) in enumerate(zip(pyt_hiddens, my_hiddens)):
            maybe_compare(pyt_hid, my_hid , "hx_"+str(i), test_case)
    except ValueError:
        maybe_compare(pyt_hiddens, my_hiddens , "hx_0", test_case)
        
        
    maybe_compare(pyt_inp.grad, my_inp.grad, "inp.grad", test_case)

print("Test passed.")
