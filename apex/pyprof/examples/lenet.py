#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.profiler as profiler
import torch.optim as optim

from apex import pyprof
pyprof.nvtx.init()

class LeNet5(nn.Module):
	def __init__(self):
		super(LeNet5, self).__init__()
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# Max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# If the size is a square you can only specify a single number
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

with torch.autograd.profiler.emit_nvtx():

	net = LeNet5().cuda()

	input = torch.randn(1, 1, 32, 32).cuda()
	out = net(input)

	target = torch.randn(10)			# a dummy target, for example
	target = target.view(1, -1).cuda()	# make it the same shape as output
	criterion = nn.MSELoss()

	# create your optimizer
	optimizer = optim.SGD(net.parameters(), lr=0.01)

	# in your training loop:
	optimizer.zero_grad()	# zero the gradient buffers

	profiler.start()
	output = net(input)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()	# Does the update
	profiler.stop()

