import unittest

from apex import amp
import random
import torch
from torch import nn

from utils import common_init, HALF

class TestRnnCells(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=True)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def run_cell_test(self, cell, state_tuple=False):
        shape = (self.b, self.h)
        for typ in [torch.float, torch.half]:
            xs = [torch.randn(shape, dtype=typ).requires_grad_()
                  for _ in range(self.t)]
            hidden_fn = lambda: torch.zeros(shape, dtype=typ)
            if state_tuple:
                hidden = (hidden_fn(), hidden_fn())
            else:
                hidden = hidden_fn()
            outputs = []
            for i in range(self.t):
                hidden = cell(xs[i], hidden)
                if state_tuple:
                    output = hidden[0]
                else:
                    output = hidden
                outputs.append(output)
            for y in outputs:
                self.assertEqual(y.type(), HALF)
            outputs[-1].float().sum().backward()
            for i, x in enumerate(xs):
                self.assertEqual(x.grad.dtype, x.dtype)

    def test_rnn_cell_is_half(self):
        cell = nn.RNNCell(self.h, self.h)
        self.run_cell_test(cell)

    def test_gru_cell_is_half(self):
        cell = nn.GRUCell(self.h, self.h)
        self.run_cell_test(cell)

    def test_lstm_cell_is_half(self):
        cell = nn.LSTMCell(self.h, self.h)
        self.run_cell_test(cell, state_tuple=True)

class TestRnns(unittest.TestCase):
    def setUp(self):
        self.handle = amp.init(enabled=True)
        common_init(self)

    def tearDown(self):
        self.handle._deactivate()

    def run_rnn_test(self, rnn, layers, bidir, state_tuple=False):
        for typ in [torch.float, torch.half]:
            x = torch.randn((self.t, self.b, self.h), dtype=typ).requires_grad_()
            hidden_fn = lambda: torch.zeros((layers + (layers * bidir),
                                             self.b, self.h), dtype=typ)
            if state_tuple:
                hidden = (hidden_fn(), hidden_fn())
            else:
                hidden = hidden_fn()
            output, _ = rnn(x, hidden)
            self.assertEqual(output.type(), HALF)
            output[-1, :, :].float().sum().backward()
            self.assertEqual(x.grad.dtype, x.dtype)

    def test_rnn_is_half(self):
        configs = [(1, False), (2, False), (2, True)]
        for layers, bidir in configs:
            rnn = nn.RNN(input_size=self.h, hidden_size=self.h, num_layers=layers,
                         nonlinearity='relu', bidirectional=bidir)
            self.run_rnn_test(rnn, layers, bidir)

    def test_gru_is_half(self):
        configs = [(1, False), (2, False), (2, True)]
        for layers, bidir in configs:
            rnn = nn.GRU(input_size=self.h, hidden_size=self.h, num_layers=layers,
                         bidirectional=bidir)
            self.run_rnn_test(rnn, layers, bidir)

    def test_lstm_is_half(self):
        configs = [(1, False), (2, False), (2, True)]
        for layers, bidir in configs:
            rnn = nn.LSTM(input_size=self.h, hidden_size=self.h, num_layers=layers,
                         bidirectional=bidir)
            self.run_rnn_test(rnn, layers, bidir, state_tuple=True)

    def test_rnn_packed_sequence(self):
        num_layers = 2
        rnn = nn.RNN(input_size=self.h, hidden_size=self.h, num_layers=num_layers)
        for typ in [torch.float, torch.half]:
            x = torch.randn((self.t, self.b, self.h), dtype=typ).requires_grad_()
            lens = sorted([random.randint(self.t // 2, self.t) for _ in range(self.b)],
                          reverse=True)
            # `pack_padded_sequence` breaks if default tensor type is non-CPU
            torch.set_default_tensor_type(torch.FloatTensor)
            lens = torch.tensor(lens, dtype=torch.int64, device=torch.device('cpu'))
            packed_seq = nn.utils.rnn.pack_padded_sequence(x, lens)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            hidden = torch.zeros((num_layers, self.b, self.h), dtype=typ)
            output, _ = rnn(packed_seq, hidden)
            self.assertEqual(output.data.type(), HALF)
            output.data.float().sum().backward()
            self.assertEqual(x.grad.dtype, x.dtype)

if __name__ == '__main__':
    unittest.main()
