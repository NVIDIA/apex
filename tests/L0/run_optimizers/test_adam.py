import copy
import math
import random
import unittest

import torch
import torch.nn.functional as F
from torch import nn
from torch.testing._internal.common_device_type import largeTensorTest

try:
    import apex
except ImportError as e:
    HAS_APEX = False
else:
    HAS_APEX = True


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.reshape(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y 


@unittest.skipIf(not HAS_APEX, "`apex` is not found.")
class AdamTest(unittest.TestCase):
    def setUp(self, seed=0):
        super().setUp()
        torch.manual_seed(seed)

        self.model = Model().cuda()
        self.model_ = Model().cuda()
        self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

        self.lr = 0.00001
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def testGradScaler(self):
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = apex.optimizers.FusedAdam(params_, lr=self.lr, capturable=False)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler_ = torch.cuda.amp.GradScaler(enabled=True)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            with torch.cuda.amp.autocast(enabled=True):
                y = self.model(x)
                loss = ((gt - y) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            # DUT
            with torch.cuda.amp.autocast(enabled=True):
                y = self.model_(x)
                loss_ = ((gt_ - y) ** 2).mean()

            scaler_.scale(loss_).backward()
            scaler_.step(optimizer_)
            scaler_.update()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(m.weight, m_.weight, atol=1e-3, rtol=1e-3, equal_nan=True)
                    torch.testing.assert_close(m.weight.grad, m_.weight.grad, atol=1e-3, rtol=1e-3, equal_nan=True)

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))
    
    def testGradScalerCapturable(self):
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = apex.optimizers.FusedAdam(params_, lr=self.lr, capturable=True)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler_ = torch.cuda.amp.GradScaler(enabled=True)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            with torch.cuda.amp.autocast(enabled=True):
                y = self.model(x)
                loss = ((gt - y) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            # DUT
            with torch.cuda.amp.autocast(enabled=True):
                y = self.model_(x)
                loss_ = ((gt_ - y) ** 2).mean()

            scaler_.scale(loss_).backward()
            scaler_.step(optimizer_)
            scaler_.update()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(m.weight, m_.weight, atol=1e-3, rtol=1e-3, equal_nan=True)
                    torch.testing.assert_close(m.weight.grad, m_.weight.grad, atol=1e-3, rtol=1e-3, equal_nan=True)

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def testGradScalerCapturableMaster(self):
        # Cast conv layers to FP16
        for m in self.model_.modules():
            if m.__class__ in [torch.nn.Conv2d]:
                m.half()
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = apex.optimizers.FusedAdam(params_, lr=self.lr, capturable=True, master_weights=True)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler_ = torch.cuda.amp.GradScaler(enabled=True)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            with torch.cuda.amp.autocast(enabled=True):
                y = self.model(x)
                loss = ((gt - y) ** 2).mean()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # DUT
            with torch.cuda.amp.autocast(enabled=True):
                y = self.model_(x)
                loss_ = ((gt_ - y) ** 2).mean()

            scaler_.scale(loss_).backward()
            scaler_.step(optimizer_)
            scaler_.update()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(m.weight, m_.weight.float(), atol=1e-3, rtol=1e-3, equal_nan=True)
                    torch.testing.assert_close(m.weight.grad, m_.weight.grad.float(), atol=1e-3, rtol=1e-3, equal_nan=True)

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()

            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def testNative(self):
        params_ = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer_ = apex.optimizers.FusedAdam(params_, lr=self.lr, capturable=False)

        for i in range(100):
            x = torch.rand([32, 1, 28, 28]).cuda().to(memory_format=torch.channels_last)
            x_ = x.clone()
            gt = torch.rand([32, 10]).cuda()
            gt_ = gt.clone()

            # Reference
            y = self.model(x)
            loss = ((gt - y) ** 2).mean()

            loss.backward()
            self.optimizer.step()
            
            # DUT
            y = self.model_(x)
            loss_ = ((gt_ - y) ** 2).mean()

            loss_.backward()
            optimizer_.step()

            for module in zip(self.model.modules(), self.model_.modules()):
                m = module[0]
                m_ = module[1]
                if isinstance(m, nn.Conv2d) or isinstance(m_, nn.Linear):
                    torch.testing.assert_close(m.weight, m_.weight, atol=1e-3, rtol=1e-3, equal_nan=True)
                    torch.testing.assert_close(m.weight.grad, m_.weight.grad, atol=1e-3, rtol=1e-3, equal_nan=True)

            # Init for next iteration
            self.optimizer.zero_grad()
            optimizer_.zero_grad()
            
            self.model_.load_state_dict(copy.deepcopy(self.model.state_dict()))

    @largeTensorTest('60GB', 'cuda')
    def testLargeTensor(self):
        t = torch.zeros(2359332864, dtype=torch.half, device='cuda')
        t2 = torch.zeros(2359332864, dtype=torch.half, device='cuda')
        grad = torch.randn_like(t)
        t.grad = grad
        t2.grad = grad
        params = [t]
        params2 = [t2]
        optimizer = apex.optimizers.FusedAdam(params, lr=self.lr)
        optimizer.step()
        optimizer2 = torch.optim.Adam(params2, lr=self.lr)
        torch.testing.assert_close(t, t2)
        torch.cuda.synchronize()


if __name__ == '__main__':
    unittest.main()

