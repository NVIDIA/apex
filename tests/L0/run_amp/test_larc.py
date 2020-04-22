import unittest

import torch
from torch import nn
from torch.nn import Parameter

from apex import amp
from apex.parallel.LARC import LARC
from utils import common_init


class MyModel(torch.nn.Module):
    def __init__(self, unique):
        super(MyModel, self).__init__()
        self.weight0 = Parameter(
            unique + torch.arange(2, device="cuda", dtype=torch.float32)
        )

    def forward(self, input):
        return (input * self.weight0).sum()


class TestLARC(unittest.TestCase):
    def setUp(self):
        self.x = torch.ones((2), device="cuda", dtype=torch.float32)
        common_init(self)

    def tearDown(self):
        pass

    def test_larc_mixed_precision(self):
        for opt_level in ["O0", "O1", "O2", "O3"]:
            model = MyModel(1)

            optimizer = LARC(
                torch.optim.SGD(
                    [{"params": model.parameters(), "lr": 0.25}], momentum=0.125
                )
            )

            model, optimizer = amp.initialize(
                model, optimizer, opt_level=opt_level, verbosity=0
            )

            optimizer.zero_grad()
            loss = model(self.x)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
