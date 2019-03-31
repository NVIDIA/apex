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


class TestMultipleModelsOptimizersLosses(unittest.TestCase):
    def setUp(self):
        self.x = torch.ones((2), device='cuda', dtype=torch.float32)
        common_init(self)

    def tearDown(self):
        pass

    def test_2models2losses1optimizer(self):
        model0 = MyModel(1)
        model1 = MyModel(2)

        optimizer = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25},
                                     {'params' : model1.parameters(), 'lr' : 0.5}],
                                    momentum=0.125)

        reference_grads = []
        for i in range(2):
            optimizer.zero_grad()
            loss0 = model0(self.x)
            loss1 = model1(self.x)
            loss0.backward()
            loss1.backward()

            reference_grads.append([param.grad.data.clone() for param in model0.parameters()] +
                                   [param.grad.data.clone() for param in model1.parameters()])

            optimizer.step()

        for opt_level in ("O0", "O1", "O2", "O3"):
          for how_to_zero in ("none", "model", "optimizer"):
            for use_multiple_loss_scalers in (True, False):
              if use_multiple_loss_scalers:
                  num_losses = 2
                  loss_ids = [0, 1]
              else:
                  num_losses = 1
                  loss_ids = [0, 0]
              
              model0 = MyModel(1)
              model1 = MyModel(2)

              models = [model0, model1]

              optimizer = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25},
                                           {'params' : model1.parameters(), 'lr' : 0.5}],
                                          momentum=0.125)

              _amp_state.allow_incoming_model_not_fp32 = True
              [model0, model1], optimizer = amp.initialize(
                  [model0, model1],
                  optimizer,
                  opt_level=opt_level,
                  verbosity=0,
                  cast_model_type=False,
                  num_losses=num_losses)
              _amp_state.allow_incoming_model_not_fp32 = False

              _amp_state.loss_scalers[0]._loss_scale = 4.0
              if use_multiple_loss_scalers:
                  _amp_state.loss_scalers[1]._loss_scale = 8.0

              for i in range(2):
                  if how_to_zero == "none":
                      for model in models:
                          for param in model.parameters():
                              param.grad = None
                  elif how_to_zero == "model":
                      for model in models:
                          model.zero_grad()
                  else:
                      optimizer.zero_grad()

                  loss0 = model0(self.x)
                  loss1 = model1(self.x)

                  with amp.scale_loss(loss0, optimizer, loss_id=loss_ids[0]) as scaled_loss:
                      scaled_loss.backward()
                  with amp.scale_loss(loss1, optimizer, loss_id=loss_ids[1]) as scaled_loss:
                      scaled_loss.backward()

                  for param, reference_grad in zip(amp.master_params(optimizer), reference_grads[i]):
                      self.assertTrue(torch.allclose(param.grad.float(), reference_grad.float()))

                  optimizer.step()

              if opt_level == "O1":
                  _amp_state.handle._deactivate()

    def test_3models2losses1optimizer(self):

        model0 = MyModel(1)
        model1 = MyModel(2)
        model2 = MyModel(3)

        optimizer = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25},
                                     {'params' : model1.parameters(), 'lr' : 0.5},
                                     {'params' : model2.parameters(), 'lr' : 0.125}],
                                     momentum=0.125)

        reference_grads = []
        for i in range(2):
            optimizer.zero_grad()
            loss0 = model0(self.x) + model2(self.x)
            loss1 = model1(self.x) + model2(self.x)
            loss0.backward()
            loss1.backward()

            reference_grads.append([param.grad.data.clone() for param in model0.parameters()] +
                                   [param.grad.data.clone() for param in model1.parameters()] +
                                   [param.grad.data.clone() for param in model2.parameters()])

            optimizer.step()

        for opt_level in ("O0", "O1", "O2", "O3"):
          for how_to_zero in ("none", "model", "optimizer"):
            for use_multiple_loss_scalers in (True, False):
              if use_multiple_loss_scalers:
                  num_losses = 2
                  loss_ids = [0, 1]
              else:
                  num_losses = 1
                  loss_ids = [0, 0]
              
              model0 = MyModel(1)
              model1 = MyModel(2)
              model2 = MyModel(3)

              models = [model0, model1, model2]

              optimizer = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25},
                                           {'params' : model1.parameters(), 'lr' : 0.5},
                                           {'params' : model2.parameters(), 'lr' : 0.125}],
                                           momentum=0.125)

              _amp_state.allow_incoming_model_not_fp32 = True
              [model0, model1, model2], optimizer = amp.initialize(
                  [model0, model1, model2],
                  optimizer,
                  opt_level=opt_level,
                  verbosity=0,
                  cast_model_type=False,
                  num_losses=num_losses)
              _amp_state.allow_incoming_model_not_fp32 = False

              _amp_state.loss_scalers[0]._loss_scale = 4.0
              if use_multiple_loss_scalers:
                  _amp_state.loss_scalers[1]._loss_scale = 8.0

              for i in range(2):
                  if how_to_zero == "none":
                      for model in models:
                          for param in model.parameters():
                              param.grad = None
                  elif how_to_zero == "model":
                      for model in models:
                          model.zero_grad()
                  else:
                      optimizer.zero_grad()

                  loss0 = model0(self.x) + model2(self.x)
                  loss1 = model1(self.x) + model2(self.x)

                  with amp.scale_loss(loss0, optimizer, loss_id=loss_ids[0]) as scaled_loss:
                      scaled_loss.backward()
                  with amp.scale_loss(loss1, optimizer, loss_id=loss_ids[1]) as scaled_loss:
                      scaled_loss.backward()

                  for param, reference_grad in zip(amp.master_params(optimizer), reference_grads[i]):
                      self.assertTrue(torch.allclose(param.grad.float(), reference_grad.float()))

                  optimizer.step()

              if opt_level == "O1":
                  _amp_state.handle._deactivate()

    def test_2models2losses2optimizers(self):
        model0 = MyModel(1)
        model1 = MyModel(2)

        optimizer0 = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25}],
                                      momentum=0.125)
        optimizer1 = torch.optim.SGD([{'params' : model1.parameters(), 'lr' : 0.5}],
                                      momentum=0.25)

        reference_grads = []
        for i in range(2):
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            loss0 = model0(self.x)
            loss1 = model1(self.x)
            loss0.backward()
            loss1.backward()

            reference_grads.append([param.grad.data.clone() for param in model0.parameters()] +
                                   [param.grad.data.clone() for param in model1.parameters()])

            optimizer0.step()
            optimizer1.step()

        for opt_level in ("O0", "O1", "O2", "O3"):
          for how_to_zero in ("none", "model", "optimizer"):
            for use_multiple_loss_scalers in (True, False):
              if use_multiple_loss_scalers:
                  num_losses = 2
                  loss_ids = [0, 1]
              else:
                  num_losses = 1
                  loss_ids = [0, 0]
              
              model0 = MyModel(1)
              model1 = MyModel(2)

              models = [model0, model1]

              optimizer0 = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25}],
                                            momentum=0.125)
              optimizer1 = torch.optim.SGD([{'params' : model1.parameters(), 'lr' : 0.5}],
                                            momentum=0.25)

              _amp_state.allow_incoming_model_not_fp32 = True
              [model0, model1], [optimizer0, optimizer1] = amp.initialize(
                  [model0, model1],
                  [optimizer0, optimizer1],
                  opt_level=opt_level,
                  verbosity=0,
                  cast_model_type=False,
                  num_losses=num_losses)
              _amp_state.allow_incoming_model_not_fp32 = False

              _amp_state.loss_scalers[0]._loss_scale = 4.0
              if use_multiple_loss_scalers:
                  _amp_state.loss_scalers[1]._loss_scale = 8.0

              for i in range(2):
                  if how_to_zero == "none":
                      for model in models:
                          for param in model.parameters():
                              param.grad = None
                  elif how_to_zero == "model":
                      for model in models:
                          model.zero_grad()
                  else:
                      optimizer0.zero_grad()
                      optimizer1.zero_grad()

                  loss0 = model0(self.x)
                  loss1 = model1(self.x)

                  with amp.scale_loss(loss0, optimizer0, loss_id=loss_ids[0]) as scaled_loss:
                      scaled_loss.backward()
                  with amp.scale_loss(loss1, optimizer1, loss_id=loss_ids[1]) as scaled_loss:
                      scaled_loss.backward()

                  master_params = list(amp.master_params(optimizer0)) + \
                                  list(amp.master_params(optimizer1))
                  for param, reference_grad in zip(master_params, reference_grads[i]):
                      self.assertTrue(torch.allclose(param.grad.float(), reference_grad.float()))

                  optimizer0.step()
                  optimizer1.step()

              if opt_level == "O1":
                  _amp_state.handle._deactivate()

    def test_3models2losses2optimizers(self):
        model0 = MyModel(1)
        model1 = MyModel(2)
        model2 = MyModel(3)

        optimizer0 = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25},
                                      {'params' : model1.parameters(), 'lr' : 1.0}],
                                     momentum=0.5)
        optimizer1 = torch.optim.SGD([{'params' : model2.parameters(), 'lr' : 0.5}],
                                     momentum=0.25)

        reference_grads = []
        for i in range(2):
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            loss0 = model0(self.x) + model1(self.x)
            loss1 = model2(self.x) + model1(self.x)
            loss0.backward()
            loss1.backward()

            reference_grads.append([param.grad.data.clone() for param in model0.parameters()] +
                                   [param.grad.data.clone() for param in model1.parameters()])

            optimizer0.step()
            optimizer1.step()

        for opt_level in ("O0", "O1", "O2", "O3"):
          for how_to_zero in ("none", "model", "optimizer"):
            for use_multiple_loss_scalers in (True, False):
              if use_multiple_loss_scalers:
                  num_losses = 2
                  loss_ids = [0, 1]
              else:
                  num_losses = 1
                  loss_ids = [0, 0]
              
              model0 = MyModel(1)
              model1 = MyModel(2)
              model2 = MyModel(3)

              models = [model0, model1, model2]

              optimizer0 = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25},
                                            {'params' : model1.parameters(), 'lr' : 1.0}],
                                           momentum=0.5)
              optimizer1 = torch.optim.SGD([{'params' : model2.parameters(), 'lr' : 0.5}],
                                           momentum=0.25)

              _amp_state.allow_incoming_model_not_fp32 = True
              [model0, model1, model2], [optimizer0, optimizer1] = amp.initialize(
                  [model0, model1, model2],
                  [optimizer0, optimizer1],
                  opt_level=opt_level,
                  verbosity=0,
                  cast_model_type=False,
                  num_losses=num_losses)
              _amp_state.allow_incoming_model_not_fp32 = False

              _amp_state.loss_scalers[0]._loss_scale = 4.0
              if use_multiple_loss_scalers:
                  _amp_state.loss_scalers[1]._loss_scale = 8.0

              for i in range(2):
                  if how_to_zero == "none":
                      for model in models:
                          for param in model.parameters():
                              param.grad = None
                  elif how_to_zero == "model":
                      for model in models:
                          model.zero_grad()
                  else:
                      optimizer0.zero_grad()
                      optimizer1.zero_grad()

                  loss0 = model0(self.x) + model1(self.x)
                  loss1 = model2(self.x) + model1(self.x)

                  with amp.scale_loss(loss0, optimizer0, loss_id=loss_ids[0]) as scaled_loss:
                      scaled_loss.backward()
                  with amp.scale_loss(loss1, [optimizer0, optimizer1], loss_id=loss_ids[1]) as scaled_loss:
                      scaled_loss.backward()

                  master_params = list(amp.master_params(optimizer0)) + \
                                  list(amp.master_params(optimizer1))
                  for param, reference_grad in zip(master_params, reference_grads[i]):
                      self.assertTrue(torch.allclose(param.grad.float(), reference_grad.float()))

                  optimizer0.step()
                  optimizer1.step()

              if opt_level == "O1":
                  _amp_state.handle._deactivate()
   
if __name__ == '__main__':
    unittest.main()
