import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from apex import amp


from utils import common_init, FLOAT


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(6)
        self.param = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = x * self.param
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        return x


class TestCheckpointing(unittest.TestCase):
    def setUp(self):
        self.initial_lr = 1e-3
        self.test_opt_levels = ("O0", "O1", "O2", "O3")

    def seed(self):
        torch.manual_seed(2809)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def check_state_dict_fp32(self, state_dict):
        for key in state_dict:
            if 'num_batches_tracked' in key:
                continue
            param = state_dict[key]
            self.assertEqual(param.type(), FLOAT,
                             'Parameter in state_dict not FLOAT')

    def train_step(self, model, optimizer, data, num_losses):
        optimizer.zero_grad()        

        output = model(data)

        # Call backward for num_losses-1
        for idx in range(1, num_losses):
            loss = output.mean() * float(idx)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward(retain_graph=True)

        # Final backward
        loss = output.mean() * float(num_losses)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()
        return output

    def compare_models(self, modelA, modelB):
        state_dictA = modelA.state_dict()
        state_dictB = modelB.state_dict()
        self.assertEqual(len(state_dictA), len(state_dictB),
                         'state_dicts have different lengths')
        for key in state_dictA:
            paramA = state_dictA[key]
            paramB = state_dictB[key]
            self.assertEqual(paramA, paramB,
                             'Parameters in state_dicts not equal')

    def test_restoring(self):
        nb_epochs = 10
        nb_epochs_restore = nb_epochs // 2
        for opt_level in self.test_opt_levels:
            for res_opt_level in self.test_opt_levels:
                for amp_before_load in [True, False]:
                    for num_losses in range(1, 3):
                        print('#' * 75 + '\n' + \
                              f'opt_level {opt_level}\n' + \
                              f'restore_opt_level {res_opt_level}\n' + \
                              f'amp_before_load {amp_before_load}\n' + \
                              f'num_losses {num_losses}\n')

                        self.seed()
                        restore_model_initialized = False

                        # Create reference model
                        model = MyModel().to('cuda')

                        optimizer = optim.SGD(model.parameters(),
                                              lr=self.initial_lr)

                        model, optimizer = amp.initialize(
                            model, optimizer, opt_level=opt_level, num_losses=num_losses)

                        # train for nb_epochs and restore after nb_epochs_restore
                        for epoch in range(nb_epochs):

                            x = torch.randn(16, 3, 24, 24, device='cuda')
                            output = self.train_step(model, optimizer, x, num_losses)
                            # Initialize model one step before comparing.
                            # Otherwise the batchnorm layers will be updated 
                            # additionally in restore_model
                            if epoch == (nb_epochs_restore - 1):
                                # Load model and optimizer
                                if not restore_model_initialized:
                                    checkpoint = {
                                        'model': model.state_dict(),
                                        'optimizer': optimizer.state_dict(),
                                        'amp': amp.state_dict()
                                    }
                                    # Check state_dict for FP32 tensors
                                    self.check_state_dict_fp32(checkpoint['model'])

                                    # Restore model
                                    restore_model = MyModel().to('cuda')
                                    restore_optimizer = optim.SGD(
                                        restore_model.parameters(),
                                        lr=self.initial_lr)

                                    if amp_before_load:
                                        restore_model, restore_optimizer = amp.initialize(
                                            restore_model,
                                            restore_optimizer,
                                            opt_level=res_opt_level,
                                            num_losses=num_losses)

                                    restore_model.load_state_dict(checkpoint['model'])
                                    restore_optimizer.load_state_dict(checkpoint['optimizer'])
                                    # FIXME: We cannot test the amp.state_dict in the same script
                                    # amp.load_state_dict(checkpoint['amp'])

                                    if not amp_before_load:
                                        restore_model, restore_optimizer = amp.initialize(
                                            restore_model,
                                            restore_optimizer,
                                            opt_level=res_opt_level,
                                            num_losses=num_losses)

                                elif epoch >= nb_epochs_restore:
                                    restore_output = self.train_step(
                                        restore_model,
                                        restore_optimizer,
                                        x,
                                        num_losses)

                                    self.assertEqual(
                                        output,
                                        restore_output,
                                        'Output of reference and restored models differ')
                                    self.compare_models(
                                        model,
                                        restore_model,
                                        'Parameters of reference and restored models differ')

    def test_loss_scale_decrease(self):
        num_losses = 3
        nb_decrease_loss_scales = [0, 1, 2]
        for opt_level in self.test_opt_levels:
            print('#' * 75 + f'\n opt_level {opt_level}\n')
            # Create new tmp copy for this run
            nb_decrease_loss_scales_tmp = list(nb_decrease_loss_scales)

            model = MyModel().to('cuda')
        
            optimizer = optim.SGD(model.parameters(),
                                  lr=1e-3)#self.initial_lr)
        
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=opt_level, num_losses=num_losses)

            if amp._amp_state.opt_properties.loss_scale != 'dynamic':
                print('Static loss scale set. Skipping opt_level.')
                continue
        
            # force to skip some updates to decrease the loss_scale
            initial_loss_scales = []
            for idx in range(num_losses):
                initial_loss_scales.append(
                    amp._amp_state.loss_scalers[idx].loss_scale())
            
            for _ in range(max(nb_decrease_loss_scales)):
                optimizer.zero_grad()
                x = torch.randn(16, 3, 24, 24, device='cuda')
                for idx in range(num_losses):
                    if nb_decrease_loss_scales_tmp[idx] > 0:
                        output = model(x * 2**17)
                        nb_decrease_loss_scales_tmp[idx] -= 1
                    else:
                        output = model(x)
                    loss = output.mean()            
                    
                    with amp.scale_loss(loss, optimizer, loss_id=idx) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                optimizer.step()
                
            # Check loss scales afterwards
            updated_loss_scales = []
            for idx in range(num_losses):
                updated_loss_scales.append(
                    amp._amp_state.loss_scalers[idx].loss_scale())
            for factor, update_ls, init_ls in zip(nb_decrease_loss_scales,
                                                  updated_loss_scales,
                                                  initial_loss_scales):
                print(update_ls == init_ls / 2**factor)
                self.assertEqual(update_ls, init_ls / 2**factor)

if __name__=='__main__':
    unittest.main()
        