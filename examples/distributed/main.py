from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from apex.fp16_utils import to_python_float

#=====START: ADDED FOR DISTRIBUTED======
'''Add custom module for distributed'''

try:
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

'''Import distributed data loader'''
import torch.utils.data
import torch.utils.data.distributed

'''Import torch.distributed'''
import torch.distributed as dist

#=====END:   ADDED FOR DISTRIBUTED======

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

#======START: ADDED FOR DISTRIBUTED======
'''
Add some distributed options. For explanation of dist-url and dist-backend please see
http://pytorch.org/tutorials/intermediate/dist_tuto.html

--local_rank will be supplied by the Pytorch launcher wrapper (torch.distributed.launch)
'''
parser.add_argument("--local_rank", default=0, type=int)

#=====END:   ADDED FOR DISTRIBUTED======

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#======START: ADDED FOR DISTRIBUTED======
'''Add a convenience flag to see if we are running distributed'''
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    '''Check that we are running with cuda, as distributed is only supported for cuda.'''
    assert args.cuda, "Distributed mode requires running with CUDA."

    '''
    Set cuda device so everything is done on the right GPU.
    THIS MUST BE DONE AS SOON AS POSSIBLE.
    '''
    torch.cuda.set_device(args.local_rank)

    '''Initialize distributed communication'''
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

#=====END:   ADDED FOR DISTRIBUTED======

torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#=====START: ADDED FOR DISTRIBUTED======
'''
Change sampler to distributed if running distributed.
Shuffle data loader only if distributed.
'''
train_dataset = datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
else:
    train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, sampler=train_sampler,
    batch_size=args.batch_size, shuffle=(train_sampler is None), **kwargs
)

#=====END:   ADDED FOR DISTRIBUTED======

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
if args.cuda:
    model.cuda()

#=====START: ADDED FOR DISTRIBUTED======
'''
Wrap model in our version of DistributedDataParallel.
This must be done AFTER the model is converted to cuda.
'''

if args.distributed:
    model = DDP(model)
#=====END:   ADDED FOR DISTRIBUTED======

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and args.local_rank == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), to_python_float(loss.data)))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader: 
        with torch.no_grad():
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += to_python_float(F.nll_loss(output, target, size_average=False).data) # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).float().cpu().sum()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    #=====START: ADDED FOR DISTRIBUTED======
    if args.distributed:
        train_sampler.set_epoch(epoch)
    #=====END:   ADDED FOR DISTRIBUTED======

    train(epoch)
    if args.local_rank == 0:
        test()
