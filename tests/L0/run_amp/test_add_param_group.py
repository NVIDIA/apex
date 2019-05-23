import unittest

import functools as ft
import itertools as it

from apex import amp
from apex.amp import _amp_state
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

from utils import common_init, HALF, FLOAT,\
    ALWAYS_HALF, ALWAYS_FLOAT, MATCH_INPUT

class MyModel(torch.nn.Module):
    def __init__(self, unique):
        super(MyModel, self).__init__()
        self.weight0 = Parameter(unique +
            torch.arange(2, device='cuda', dtype=torch.float32))
        self.weight1 = Parameter(1. + unique + torch.arange(2, device='cuda', dtype=torch.float16))

    @staticmethod
    def ops(input, weight0, weight1):
        return ((input*(weight0.float()))*(weight1.float())).sum()

    def forward(self, input):
        return self.ops(input, self.weight0, self.weight1)


# Abandon all hope, ye who enter here.


class TestAddParamGroup(unittest.TestCase):
    def setUp(self):
        self.x = torch.ones((2), device='cuda', dtype=torch.float32)
        common_init(self)

    def tearDown(self):
        pass

    def zero_grad(self, models, optimizer, how_to_zero):
        if how_to_zero == "none":
            for model in models:
                for param in model.parameters():
                    param.grad = None
        elif how_to_zero == "model":
            for model in models:
                model.zero_grad()
        elif how_to_zero == "optimizer":
            optimizer.zero_grad()

    def test_add_param_group(self):
        for opt_level in ("O0", "O1", "O2", "O3"):
          for zero_before_add in (True, False):
            for try_accumulation in (True, False):
              model0 = MyModel(1)
              model1 = MyModel(2)

              optimizer = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25}],
                                          momentum=0.125)

              optimizer.zero_grad()
              loss = model0(self.x)
              loss.backward()
              optimizer.step()

              if zero_before_add:
                  optimizer.zero_grad()
              optimizer.add_param_group({'params' : model1.parameters(), 'lr' : 0.5})
              if not zero_before_add:
                  optimizer.zero_grad()

              loss = model0(self.x) + model1(self.x)
              loss.backward(retain_graph=try_accumulation)
              if try_accumulation:
                  loss.backward()
              optimizer.step()

              # Once more to make sure the new params pick up momemtums properly
              optimizer.zero_grad()
              loss = model0(self.x) + model1(self.x)
              loss.backward(retain_graph=try_accumulation)
              if try_accumulation:
                  loss.backward()
              optimizer.step()

              reference_params = [param.data.clone() for param in model0.parameters()] + \
                                 [param.data.clone() for param in model1.parameters()]

              for how_to_zero in "none", "model", "optimizer":
                  model0 = MyModel(1)
                  model1 = MyModel(2)

                  optimizer = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25}],
                                              momentum=0.125)

                  _amp_state.allow_incoming_model_not_fp32 = True
                  [model0, model1], optimizer = amp.initialize([model0, model1],
                      optimizer,
                      opt_level=opt_level,
                      verbosity=0,
                      cast_model_type=False)
                  _amp_state.allow_incoming_model_not_fp32 = False

                  _amp_state.loss_scalers[0]._loss_scale = 4.0

                  self.zero_grad([model0, model1], optimizer, how_to_zero)
                  loss = model0(self.x)
                  with amp.scale_loss(loss, optimizer) as scaled_loss:
                      scaled_loss.backward()
                  optimizer.step()

                  if zero_before_add:
                      self.zero_grad([model0, model1], optimizer, how_to_zero)
                  optimizer.add_param_group({'params' : model1.parameters(), 'lr' : 0.5})
                  if not zero_before_add:
                      self.zero_grad([model0, model1], optimizer, how_to_zero)

                  loss = model0(self.x) + model1(self.x)
                  with amp.scale_loss(loss, optimizer) as scaled_loss:
                      scaled_loss.backward(retain_graph=try_accumulation)
                  if try_accumulation:
                      with amp.scale_loss(loss, optimizer) as scaled_loss:
                          scaled_loss.backward()
                  optimizer.step()

                  # Once more to make sure the new params pick up momentums properly
                  self.zero_grad([model0, model1], optimizer, how_to_zero)
                  loss = model0(self.x) + model1(self.x)
                  with amp.scale_loss(loss, optimizer) as scaled_loss:
                      scaled_loss.backward(retain_graph=try_accumulation)
                  if try_accumulation:
                      with amp.scale_loss(loss, optimizer) as scaled_loss:
                          scaled_loss.backward()
                  optimizer.step()

                  final_params = [param.data.clone() for param in model0.parameters()] + \
                                 [param.data.clone() for param in model1.parameters()]

                  for reference, final in zip(reference_params, final_params):
                      self.assertTrue(torch.allclose(reference.to(final.dtype), final),
                                      "opt_level = {}, how_to_zero = {}, zero_before_add = {}".format(
                                      opt_level, how_to_zero, zero_before_add))


if __name__ == '__main__':
    unittest.main()
