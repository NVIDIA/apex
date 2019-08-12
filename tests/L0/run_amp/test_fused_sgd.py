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


try:
  import amp_C
  disabled = False
  from apex.optimizers import FusedSGD as FusedSGD
except ImportError as err:
  print("amp_C fused kernels unavailable, disabling TestMultiTensorApply.  ImportError was ", err)
  disabled = True


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

# This is hands down the ugliest code I have ever written, but it succeeds in testing
# multiple models/optimizers/losses fairly thoroughly.  Many of the different test cases
# require slightly divergent code in a way that seems near-impossible to genericize into a simple
# cross product or nested loops.

class TestMultipleModelsOptimizersLosses(unittest.TestCase):
    def setUp(self):
        self.x = torch.ones((2), device='cuda', dtype=torch.float32)
        common_init(self)

    def tearDown(self):
        pass

    @unittest.skipIf(disabled, "amp_C is unavailable")
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

        final_params = [param.data.clone() for param in model0.parameters()] + \
                       [param.data.clone() for param in model1.parameters()]

        for materialize_master_grads in (False, True):
          for opt_level in ("O0", "O1", "O2", "O3"):
            for how_to_zero in ("none", "model", "optimizer"):
              for use_multiple_loss_scalers in (False, True):
                if opt_level == "O1" or opt_level == "O2":
                    inject_inf_iters = (-1, 0, 1)
                else:
                    inject_inf_iters = (-1,)

                for inject_inf in inject_inf_iters:
                  if inject_inf >= 0:
                     inject_inf_locs = ("fp16", "fp32")
                     which_backwards = (0, 1)
                  else:
                     inject_inf_locs = ("fdsa",)
                     which_backwards = (None,)

                  for inject_inf_loc in inject_inf_locs:
                    for which_backward in which_backwards:
                        if use_multiple_loss_scalers:
                            num_losses = 2
                            loss_ids = [0, 1]
                        else:
                            num_losses = 1
                            loss_ids = [0, 0]

                        if inject_inf >= 0:
                            iters = 3
                        else:
                            iters = 2

                        model0 = MyModel(1)
                        model1 = MyModel(2)

                        models = [model0, model1]

                        optimizer = FusedSGD([{'params' : model0.parameters(), 'lr' : 0.25},
                                              {'params' : model1.parameters(), 'lr' : 0.5}],
                                             momentum=0.125,
                                             materialize_master_grads=materialize_master_grads)

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
                            _amp_state.loss_scalers[1]._loss_scale = 16.0

                        unskipped = 0
                        for i in range(iters):
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
                                if i == inject_inf and which_backward == 0:
                                    if inject_inf_loc == "fp32":
                                        model0.weight0.grad[0] = float('inf')
                                    elif inject_inf_loc == "fp16":
                                        model0.weight1.grad[0] = float('inf')
                            with amp.scale_loss(loss1, optimizer, loss_id=loss_ids[1]) as scaled_loss:
                                scaled_loss.backward()
                                if i == inject_inf and which_backward == 1:
                                    if inject_inf_loc == "fp32":
                                        model1.weight0.grad[0] = float('inf')
                                    elif inject_inf_loc == "fp16":
                                        model1.weight1.grad[0] = float('inf')

                            if i != inject_inf:
                                master_params = amp.master_params(optimizer)
                                for param, reference_grad in zip(master_params, reference_grads[unskipped]):
                                    if opt_level == "O2" and not materialize_master_grads:
                                        continue
                                    else:
                                        self.assertTrue(torch.allclose(param.grad.float(), reference_grad.float()),
                                                        "opt_level {} i {} inject_inf {} which_backward {} inject_inf_loc {} use_multiple_loss_scalers {}".format(opt_level, i, inject_inf, which_backward, inject_inf_loc, use_multiple_loss_scalers))
                                unskipped += 1
                            optimizer.step()

                        model_params = [p for p in model0.parameters()] + [p for p in model1.parameters()]
                        for model, master, reference in zip(
                                model_params,
                                amp.master_params(optimizer),
                                final_params):
                            self.assertTrue(torch.allclose(model, reference))
                            self.assertTrue(torch.allclose(model, master.to(model.dtype)))

                        if opt_level == "O1":
                            _amp_state.handle._deactivate()

    @unittest.skipIf(disabled, "amp_C is unavailable")
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


        final_params = [param.data.clone() for param in model0.parameters()] + \
                       [param.data.clone() for param in model1.parameters()] + \
                       [param.data.clone() for param in model2.parameters()]

        for materialize_master_grads in (False, True):
          for opt_level in ("O0", "O1", "O2", "O3"):
            for how_to_zero in ("none", "model", "optimizer"):
              for use_multiple_loss_scalers in (False, True):
                if opt_level == "O1" or opt_level == "O2":
                    inject_inf_iters = (-1, 0, 1)
                else:
                    inject_inf_iters = (-1,)

                for inject_inf in inject_inf_iters:
                  if inject_inf >= 0:
                     inject_inf_locs = ("fp16", "fp32")
                     which_backwards = (0, 1)
                  else:
                     inject_inf_locs = ("fdsa",)
                     which_backwards = (None,)

                  for inject_inf_loc in inject_inf_locs:
                    for which_backward in which_backwards:
                      if use_multiple_loss_scalers:
                          num_losses = 2
                          loss_ids = [0, 1]
                      else:
                          num_losses = 1
                          loss_ids = [0, 0]

                      if inject_inf >= 0:
                          iters = 3
                          if which_backward == 0:
                              which_models = (0, 2)
                          elif which_backward == 1:
                              which_models = (1, 2)
                      else:
                          iters = 2
                          which_models = (None,)

                      for which_model in which_models:
                          model0 = MyModel(1)
                          model1 = MyModel(2)
                          model2 = MyModel(3)

                          models = [model0, model1, model2]

                          optimizer = FusedSGD([{'params' : model0.parameters(), 'lr' : 0.25},
                                               {'params' : model1.parameters(), 'lr' : 0.5},
                                               {'params' : model2.parameters(), 'lr' : 0.125}],
                                               momentum=0.125,
                                               materialize_master_grads=materialize_master_grads)

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
                              _amp_state.loss_scalers[1]._loss_scale = 16.0

                          unskipped = 0
                          for i in range(iters):
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
                                  if i == inject_inf and which_backward == 0:
                                      if which_model == 0:
                                          inj_model = model0
                                      elif which_model == 2:
                                          inj_model = model2
                                      else:
                                          raise RuntimeError(which_model + " invalid for loss 0")
                                      if inject_inf_loc == "fp32":
                                          inj_model.weight0.grad[0] = float('inf')
                                      elif inject_inf_loc == "fp16":
                                          inj_model.weight1.grad[0] = float('inf')
                              with amp.scale_loss(loss1, optimizer, loss_id=loss_ids[1]) as scaled_loss:
                                  scaled_loss.backward()
                                  if i == inject_inf and which_backward == 1:
                                      if which_model == 1:
                                          inj_model = model1
                                      elif which_model == 2:
                                          inj_model = model2
                                      else:
                                          raise RuntimeError(which_model + " invalid for loss 1 ")
                                      if inject_inf_loc == "fp32":
                                          inj_model.weight0.grad[0] = float('inf')
                                      elif inject_inf_loc == "fp16":
                                          inj_model.weight1.grad[0] = float('inf')

                              if i != inject_inf:
                                  master_params = amp.master_params(optimizer)
                                  for param, reference_grad in zip(master_params, reference_grads[unskipped]):
                                      if opt_level == "O2" and not materialize_master_grads:
                                          continue
                                      else:
                                          self.assertTrue(torch.allclose(param.grad.float(), reference_grad.float()),
                                          "opt_level {} i {} inject_inf {} which_backward {} inject_inf_loc {} which_model {} use_multiple_loss_scalers {}".format(opt_level, i, inject_inf, which_backward, inject_inf_loc, which_model, use_multiple_loss_scalers))
                                  unskipped += 1

                              optimizer.step()

                          model_params = [p for p in model0.parameters()] + \
                                         [p for p in model1.parameters()] + \
                                         [p for p in model2.parameters()]
                          for model, master, reference in zip(
                                  model_params,
                                  amp.master_params(optimizer),
                                  final_params):
                              self.assertTrue(torch.allclose(model, reference))
                              self.assertTrue(torch.allclose(model, master.to(model.dtype)))

                          if opt_level == "O1":
                              _amp_state.handle._deactivate()

    @unittest.skipIf(disabled, "amp_C is unavailable")
    def test_2models2losses2optimizers(self):
        model0 = MyModel(1)
        model1 = MyModel(2)

        optimizer0 = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25}],
                                      momentum=0.125)
        optimizer1 = torch.optim.SGD([{'params' : model1.parameters(), 'lr' : 0.5}],
                                      momentum=0.25)

        # Don't do it like this:  reference_grads = [[]]*5
        # because then it creates a list of 5 references to the same "[]" and appending
        # to any of them effectively makes you append to all of them, which multiplies
        # the resulting size of reference_grads by 5x and needless to say makes the test fail.
        reference_grads = [[], [], [], [], []]
        final_params = [None, None, None, None, None]
        for i in range(2):
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            loss0 = model0(self.x)
            loss1 = model1(self.x)
            loss0.backward()
            loss1.backward()

            reference_grads[0].append([param.grad.data.clone() for param in model0.parameters()] +
                                   [param.grad.data.clone() for param in model1.parameters()])

            optimizer0.step()
            optimizer1.step()

        final_params[0] = [param.data.clone() for param in model0.parameters()] + \
                          [param.data.clone() for param in model1.parameters()]

        def what_got_skipped(which_iter, which_backward):
            if which_iter == 0 and which_backward == 0:
                return 1
            if which_iter == 0 and which_backward == 1:
                return 2
            if which_iter == 1 and which_backward == 0:
                return 3
            if which_iter == 1 and which_backward == 1:
                return 4
            return 0

        for which_iter in (0,1):
            for which_backward in (0,1):
                model0 = MyModel(1)
                model1 = MyModel(2)

                optimizer0 = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25}],
                                              momentum=0.125)
                optimizer1 = torch.optim.SGD([{'params' : model1.parameters(), 'lr' : 0.5}],
                                              momentum=0.25)

                for i in range(3):
                    optimizer0.zero_grad()
                    optimizer1.zero_grad()
                    loss0 = model0(self.x)
                    loss1 = model1(self.x)
                    loss0.backward()
                    loss1.backward()

                    if i != which_iter:
                        reference_grads[what_got_skipped(which_iter, which_backward)].append(
                            [param.grad.data.clone() for param in model0.parameters()] +
                            [param.grad.data.clone() for param in model1.parameters()])

                    if i == which_iter:
                        if which_backward == 0:
                            optimizer1.step()
                        else:
                            optimizer0.step()
                    else:
                        optimizer0.step()
                        optimizer1.step()

                final_params[what_got_skipped(which_iter, which_backward)] = \
                    [param.data.clone() for param in model0.parameters()] + \
                    [param.data.clone() for param in model1.parameters()]

        for materialize_master_grads in (False, True):
          for opt_level in ("O0", "O1", "O2", "O3"):
            for how_to_zero in ("none", "model", "optimizer"):
              for use_multiple_loss_scalers in (False, True):
                if opt_level == "O1" or opt_level == "O2":
                    inject_inf_iters = (-1, 0, 1)
                else:
                    inject_inf_iters = (-1,)

                for inject_inf in inject_inf_iters:
                  if inject_inf >= 0:
                     inject_inf_locs = ("fp16", "fp32")
                     which_backwards = (0, 1)
                  else:
                     inject_inf_locs = ("fdsa",)
                     which_backwards = (None,)

                  for inject_inf_loc in inject_inf_locs:
                    for which_backward in which_backwards:
                        if use_multiple_loss_scalers:
                            num_losses = 2
                            loss_ids = [0, 1]
                        else:
                            num_losses = 1
                            loss_ids = [0, 0]

                        if inject_inf >= 0:
                            iters = 3
                        else:
                            iters = 2

                        model0 = MyModel(1)
                        model1 = MyModel(2)

                        models = [model0, model1]

                        optimizer0 = FusedSGD([{'params' : model0.parameters(), 'lr' : 0.25}],
                                              momentum=0.125, materialize_master_grads=materialize_master_grads)
                        optimizer1 = FusedSGD([{'params' : model1.parameters(), 'lr' : 0.5}],
                                              momentum=0.25, materialize_master_grads=materialize_master_grads)

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
                            _amp_state.loss_scalers[1]._loss_scale = 16.0

                        unskipped = 0
                        for i in range(iters):
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
                                if i == inject_inf and which_backward == 0:
                                    if inject_inf_loc == "fp32":
                                        model0.weight0.grad[0] = float('inf')
                                    elif inject_inf_loc == "fp16":
                                        model0.weight1.grad[0] = float('inf')
                            with amp.scale_loss(loss1, optimizer1, loss_id=loss_ids[1]) as scaled_loss:
                                scaled_loss.backward()
                                if i == inject_inf and which_backward == 1:
                                    if inject_inf_loc == "fp32":
                                        model1.weight0.grad[0] = float('inf')
                                    elif inject_inf_loc == "fp16":
                                        model1.weight1.grad[0] = float('inf')

                            # print("opt_level {} i {} inject_inf {} which_backward {} inject_inf_loc {} use_multiple_loss_scalers {}".format(opt_level, i, inject_inf, which_backward, inject_inf_loc, use_multiple_loss_scalers))

                            if i != inject_inf:
                                master_params = list(amp.master_params(optimizer0)) + \
                                                list(amp.master_params(optimizer1))
                                for param, reference_grad in zip(master_params,
                                        reference_grads[what_got_skipped(inject_inf, which_backward)][unskipped]):
                                    if opt_level == "O2" and not materialize_master_grads:
                                        continue
                                    else:
                                        self.assertTrue(torch.allclose(param.grad.float(), reference_grad.float()))
                                unskipped += 1

                            optimizer0.step()
                            optimizer1.step()

                        model_params = [p for p in model0.parameters()] + [p for p in model1.parameters()]
                        master_params = [p for p in amp.master_params(optimizer0)] + \
                                        [p for p in amp.master_params(optimizer1)]
                        for model, master, reference in zip(
                                model_params,
                                master_params,
                                final_params[what_got_skipped(inject_inf, which_backward)]):
                            self.assertTrue(torch.allclose(model, reference))
                            self.assertTrue(torch.allclose(model, master.to(model.dtype)))

                        if opt_level == "O1":
                            _amp_state.handle._deactivate()

    @unittest.skipIf(disabled, "amp_C is unavailable")
    def test_3models2losses2optimizers(self):
        model0 = MyModel(1)
        model1 = MyModel(2)
        model2 = MyModel(3)

        optimizer0 = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25},
                                      {'params' : model1.parameters(), 'lr' : 1.0}],
                                     momentum=0.5)
        optimizer1 = torch.optim.SGD([{'params' : model2.parameters(), 'lr' : 0.5}],
                                     momentum=0.25)

        # Again, can't do this:  reference_grads = [[]]*9
        reference_grads = [[], [], [], [], [], [], [], [], []]
        final_params = [None, None, None, None, None, None, None, None, None]
        for i in range(2):
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            loss0 = model0(self.x) + model1(self.x)
            loss1 = model2(self.x) + model1(self.x)
            loss0.backward()
            loss1.backward()

            reference_grads[0].append([param.grad.data.clone() for param in model0.parameters()] +
                                   [param.grad.data.clone() for param in model1.parameters()])

            optimizer0.step()
            optimizer1.step()

        final_params[0] = \
            [param.data.clone() for param in model0.parameters()] + \
            [param.data.clone() for param in model1.parameters()] + \
            [param.data.clone() for param in model2.parameters()]

        def what_got_skipped(which_iter, which_backward, which_model):
            if which_iter == 0:
                if which_backward == 0:
                    if which_model == 0:
                        return 1
                    if which_model == 1:
                        return 2
                if which_backward == 1:
                    if which_model == 2:
                        return 3
                    if which_model == 1:
                        return 4
            if which_iter == 1:
                if which_backward == 0:
                    if which_model == 0:
                        return 5
                    if which_model == 1:
                        return 6
                if which_backward == 1:
                    if which_model == 2:
                        return 7
                    if which_model == 1:
                        return 8
            return 0

        for which_iter in (0,1):
            for which_backward in (0,1):
                if which_backward == 0:
                    which_models = (0,1)
                if which_backward == 1:
                    which_models = (2,1)
                for which_model in which_models:

                    model0 = MyModel(1)
                    model1 = MyModel(2)
                    model2 = MyModel(3)

                    optimizer0 = torch.optim.SGD([{'params' : model0.parameters(), 'lr' : 0.25},
                                                  {'params' : model1.parameters(), 'lr' : 1.0}],
                                                 momentum=0.5)
                    optimizer1 = torch.optim.SGD([{'params' : model2.parameters(), 'lr' : 0.5}],
                                                 momentum=0.25)

                    for i in range(3):
                        optimizer0.zero_grad()
                        optimizer1.zero_grad()
                        loss0 = model0(self.x) + model1(self.x)
                        loss1 = model2(self.x) + model1(self.x)
                        loss0.backward()
                        loss1.backward()

                        if i != which_iter:
                            reference_grads[what_got_skipped(which_iter,
                                    which_backward, which_model)].append(
                                [param.grad.data.clone() for param in model0.parameters()] +
                                [param.grad.data.clone() for param in model1.parameters()])

                        if i == which_iter:
                            if which_backward == 0:
                                # if which_model == 0:
                                    optimizer1.step()
                                # if which_model == 1:
                                #     optimizer1.step()
                            if which_backward == 1:
                                # if which_model == 2:
                                #     optimizer0.step()
                                # if which_model == 1:
                                    continue
                        else:
                            optimizer0.step()
                            optimizer1.step()

                    final_params[what_got_skipped(which_iter, which_backward, which_model)] = \
                        [param.data.clone() for param in model0.parameters()] + \
                        [param.data.clone() for param in model1.parameters()] + \
                        [param.data.clone() for param in model2.parameters()]

        for materialize_master_grads in (False, True):
          for opt_level in ("O0", "O1", "O2", "O3"):
            for how_to_zero in ("none", "model", "optimizer"):
              for use_multiple_loss_scalers in (False, True):
                if opt_level == "O1" or opt_level == "O2":
                    inject_inf_iters = (-1, 0, 1)
                else:
                    inject_inf_iters = (-1,)

                for inject_inf in inject_inf_iters:
                  if inject_inf >= 0:
                     inject_inf_locs = ("fp16", "fp32")
                     which_backwards = (0, 1)
                  else:
                     inject_inf_locs = ("fdsa",)
                     which_backwards = (None,)

                  for inject_inf_loc in inject_inf_locs:
                    for which_backward in which_backwards:
                      if use_multiple_loss_scalers:
                          num_losses = 2
                          loss_ids = [0, 1]
                      else:
                          num_losses = 1
                          loss_ids = [0, 0]

                      if inject_inf >= 0:
                          iters = 3
                          if which_backward == 0:
                              which_models = (0, 1)
                          elif which_backward == 1:
                              which_models = (2, 1)
                      else:
                          iters = 2
                          which_models = (None,)

                      for which_model in which_models:
                          model0 = MyModel(1)
                          model1 = MyModel(2)
                          model2 = MyModel(3)

                          models = [model0, model1, model2]

                          optimizer0 = FusedSGD([{'params' : model0.parameters(), 'lr' : 0.25},
                                            {'params' : model1.parameters(), 'lr' : 1.0}],
                                            momentum=0.5, materialize_master_grads=materialize_master_grads)
                          optimizer1 = FusedSGD([{'params' : model2.parameters(), 'lr' : 0.5}],
                                                momentum=0.25, materialize_master_grads=materialize_master_grads)

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
                              _amp_state.loss_scalers[1]._loss_scale = 16.0

                          unskipped = 0
                          for i in range(iters):
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
                                  if i == inject_inf and which_backward == 0:
                                      if which_model == 0:
                                          inj_model = model0
                                      elif which_model == 1:
                                          inj_model = model1
                                      else:
                                          raise RuntimeError(which_model + " invalid for loss 0")
                                      if inject_inf_loc == "fp32":
                                          inj_model.weight0.grad[0] = float('inf')
                                      elif inject_inf_loc == "fp16":
                                          inj_model.weight1.grad[0] = float('inf')
                              with amp.scale_loss(loss1, [optimizer0, optimizer1], loss_id=loss_ids[1]) as scaled_loss:
                                  scaled_loss.backward()
                                  if i == inject_inf and which_backward == 1:
                                      if which_model == 2:
                                          inj_model = model2
                                      elif which_model == 1:
                                          inj_model = model1
                                      else:
                                          raise RuntimeError(which_model + " invalid for loss 1 ")
                                      if inject_inf_loc == "fp32":
                                          inj_model.weight0.grad[0] = float('inf')
                                      elif inject_inf_loc == "fp16":
                                          inj_model.weight1.grad[0] = float('inf')

                              if i != inject_inf:
                                  master_params = list(amp.master_params(optimizer0)) + \
                                                  list(amp.master_params(optimizer1))
                                  for param, reference_grad in zip(master_params,
                                        reference_grads[what_got_skipped(inject_inf,
                                            which_backward, which_model)][unskipped]):
                                      if opt_level == "O2" and not materialize_master_grads:
                                          continue
                                      else:
                                          self.assertTrue(torch.allclose(param.grad.float(), reference_grad.float()))
                                  unskipped += 1

                              optimizer0.step()
                              optimizer1.step()

                          model_params = [p for p in model0.parameters()] + \
                                         [p for p in model1.parameters()] + \
                                         [p for p in model2.parameters()]
                          master_params = [p for p in amp.master_params(optimizer0)] + \
                                          [p for p in amp.master_params(optimizer1)]

                          # print("opt_level {} i {} inject_inf {} which_backward {} inject_inf_loc {} use_multiple_loss_scalers {} which_model {}".format(opt_level, i, inject_inf, which_backward, inject_inf_loc, use_multiple_loss_scalers, which_model))

                          for model, master, reference in zip(
                                  model_params,
                                  master_params,
                                  final_params[what_got_skipped(inject_inf, which_backward, which_model)]):
                              self.assertTrue(torch.allclose(model, reference))
                              self.assertTrue(torch.allclose(model, master.to(model.dtype)))

                          if opt_level == "O1":
                              _amp_state.handle._deactivate()

if __name__ == '__main__':
    unittest.main()
