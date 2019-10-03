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

    def train_step(self, model, optimizer, data, loss_ids):
        optimizer.zero_grad()        

        output = model(data)

        # Call backward for num_losses-1
        for idx in loss_ids:
            loss = output.mean()
            with amp.scale_loss(loss, optimizer, loss_id=idx) as scaled_loss:
                scaled_loss.backward(retain_graph=True)

        optimizer.step()
        return output

    def compare_models(self, modelA, modelB, test_setup=''):
        state_dictA = modelA.state_dict()
        state_dictB = modelB.state_dict()
        self.assertEqual(len(state_dictA), len(state_dictB),
                         'state_dicts have different lengths' + test_setup)
        for key in state_dictA:
            paramA = state_dictA[key]
            paramB = state_dictB[key]
            self.assertTrue((paramA==paramB).all(),
                msg='Parameters in state_dices not equal.' +
                    'key: {}\nparam: {}\nrestored: {}\ndiff: {} for {}'.format(
                        key, paramA, paramB, paramA - paramB, test_setup))

    def test_restoring(self):
        nb_epochs = 10
        nb_epochs_restore = nb_epochs // 2
        for opt_level in self.test_opt_levels:
            for res_opt_level in self.test_opt_levels:
                for amp_before_load in [True, False]:
                    for num_losses in range(1, 3):
                        test_setup = ('#' * 75 + '\n' + \
                              f'opt_level {opt_level}\n' + \
                              f'restore_opt_level {res_opt_level}\n' + \
                              f'amp_before_load {amp_before_load}\n' + \
                              f'num_losses {num_losses}\n')

                        self.seed()

                        # Create reference model
                        model = MyModel().to('cuda')

                        optimizer = optim.SGD(model.parameters(),
                                              lr=self.initial_lr)

                        # Initialize with num_losses*2 for the original model and the restored one
                        model, optimizer = amp.initialize(
                            model, optimizer, opt_level=opt_level,
                            num_losses=num_losses*2, verbosity=0)

                        # Compare training behavior for same restore option
                        # We cannot really generalize it, since a saved model in O0
                        # would introduce a skipped step in O1, which will raise an error
                        if opt_level == res_opt_level:
                            # train for nb_epochs and restore after nb_epochs_restore
                            for epoch in range(nb_epochs):
    
                                x = torch.randn(16, 3, 24, 24, device='cuda')
                                output = self.train_step(
                                    model, optimizer, x, range(num_losses))
                                # Initialize model one step before comparing.
                                # Otherwise the batchnorm layers will be updated 
                                # additionally in restore_model
                                if epoch == (nb_epochs_restore - 1):
                                    # Load model and optimizer
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
                                            num_losses=num_losses*2,
                                            verbosity=0)

                                    restore_model.load_state_dict(checkpoint['model'])
                                    restore_optimizer.load_state_dict(checkpoint['optimizer'])
                                    # FIXME: We cannot test the amp.state_dict in the same script
                                    # amp.load_state_dict(checkpoint['amp'])

                                    if not amp_before_load:
                                        restore_model, restore_optimizer = amp.initialize(
                                            restore_model,
                                            restore_optimizer,
                                            opt_level=res_opt_level,
                                            num_losses=num_losses*2,
                                            verbosity=0)

                                elif epoch >= nb_epochs_restore:
                                    restore_output = self.train_step(
                                        restore_model,
                                        restore_optimizer,
                                        x,
                                        range(num_losses, num_losses*2))
                                    self.assertTrue(
                                        torch.allclose(output.float(), restore_output.float()),
                                        'Output of reference and restored models differ for ' + test_setup)
                                    self.compare_models(model, restore_model, test_setup)
                        # if opt_level != res_opt_level
                        else:
                            # skip tests for different opt_levels
                            continue

    def test_loss_scale_decrease(self):
        num_losses = 3
        nb_decrease_loss_scales = [0, 1, 2]
        for opt_level in self.test_opt_levels:
            #print('#' * 75 + f'\n opt_level {opt_level}\n')
            # Create new tmp copy for this run
            nb_decrease_loss_scales_tmp = list(nb_decrease_loss_scales)

            model = MyModel().to('cuda')
        
            optimizer = optim.SGD(model.parameters(),
                                  lr=self.initial_lr)
        
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=opt_level, num_losses=num_losses,
                verbosity=0)

            if amp._amp_state.opt_properties.loss_scale != 'dynamic':
                #print('Static loss scale set. Skipping opt_level.')
                continue
        
            # force to skip some updates to decrease the loss_scale
            initial_loss_scales = []
            for idx in range(num_losses):
                initial_loss_scales.append(
                    amp._amp_state.loss_scalers[idx].loss_scale())
            
            for _ in range(len(nb_decrease_loss_scales)):
                x = torch.randn(16, 3, 24, 24, device='cuda')
                for idx in range(num_losses):
                    while nb_decrease_loss_scales_tmp[idx] > 0:
                        optimizer.zero_grad()
                        output = model(x * 2**17)
                        loss = output.mean()            
                    
                        with amp.scale_loss(loss, optimizer, loss_id=idx) as scaled_loss:
                            scaled_loss.backward(retain_graph=True)
                        optimizer.step()
                        nb_decrease_loss_scales_tmp[idx] -= 1
                
            # Check loss scales afterwards
            updated_loss_scales = []
            for idx in range(num_losses):
                updated_loss_scales.append(
                    amp._amp_state.loss_scalers[idx].loss_scale())
            for factor, update_ls, init_ls in zip(nb_decrease_loss_scales,
                                                  updated_loss_scales,
                                                  initial_loss_scales):
                self.assertEqual(update_ls, init_ls / 2**factor)

            # Check state dict
            amp_state_dict = amp.state_dict()
            for scaler_idx, factor, init_ls in zip(amp_state_dict,
                                                   nb_decrease_loss_scales,
                                                   initial_loss_scales):
                scaler = amp_state_dict[scaler_idx]
                self.assertEqual(scaler['loss_scale'], init_ls / 2**factor)
                unskipped_target = 0
                self.assertEqual(scaler['unskipped'], unskipped_target)

    def test_state_dict(self):
        for opt_level in self.test_opt_levels:
            # Skip O3
            if opt_level == 'O3':
                continue

            model = MyModel().to('cuda')
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=opt_level, verbosity=0)

            # Export state_dict and check for Half
            state_dict = model.state_dict()
            for key in state_dict:
                self.assertFalse('Half' in state_dict[key].type())

            # Check, if model is still trainable
            # Create dummy data
            data = torch.randn(10, 3, 4, 4, device='cuda')
            target = torch.randn(10, 6, 4, 4, device='cuda')
            
            # Get initnial loss
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, target)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            last_loss = loss.item()

            # train for some epochs
            for epoch in range(10):
                optimizer.zero_grad()
                output = model(data)
                loss = F.mse_loss(output, target)
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                self.assertTrue(loss.item() < last_loss)
                last_loss = loss.item()

if __name__=='__main__':
    unittest.main()
        
