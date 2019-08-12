#!/usr/bin/env python3

"""
An example showing use of nested NVTX markers.
"""

import torch
import torch.nn as nn

import torch.cuda.profiler as profiler
import torch.cuda.nvtx as nvtx
from apex import pyprof
pyprof.nvtx.init()

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
	expansion = 4
	count = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

		self.id = Bottleneck.count
		Bottleneck.count += 1

	def forward(self, x):
		identity = x

		nvtx.range_push("layer:Bottleneck_{}".format(self.id))

		nvtx.range_push("layer:Conv1")
		out = self.conv1(x)
		nvtx.range_pop()

		nvtx.range_push("layer:BN1")
		out = self.bn1(out)
		nvtx.range_pop()

		nvtx.range_push("layer:ReLU")
		out = self.relu(out)
		nvtx.range_pop()

		nvtx.range_push("layer:Conv2")
		out = self.conv2(out)
		nvtx.range_pop()

		nvtx.range_push("layer:BN2")
		out = self.bn2(out)
		nvtx.range_pop()

		nvtx.range_push("layer:ReLU")
		out = self.relu(out)
		nvtx.range_pop()

		nvtx.range_push("layer:Conv3")
		out = self.conv3(out)
		nvtx.range_pop()

		nvtx.range_push("layer:BN3")
		out = self.bn3(out)
		nvtx.range_pop()

		if self.downsample is not None:
			nvtx.range_push("layer:Downsample")
			identity = self.downsample(x)
			nvtx.range_pop()

		nvtx.range_push("layer:Residual")
		out += identity
		nvtx.range_pop()

		nvtx.range_push("layer:ReLU")
		out = self.relu(out)
		nvtx.range_pop()

		nvtx.range_pop()

		return out

class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1000,
				 groups=1, width_per_group=64, norm_layer=None):
		super(ResNet, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		self.inplanes = 64
		self.dilation = 1

		self.groups = groups
		self.base_width = width_per_group
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
							self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def forward(self, x):

		nvtx.range_push("layer:conv1_x")
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		nvtx.range_pop()

		nvtx.range_push("layer:conv2_x")
		x = self.layer1(x)
		nvtx.range_pop()

		nvtx.range_push("layer:conv3_x")
		x = self.layer2(x)
		nvtx.range_pop()

		nvtx.range_push("layer:conv4_x")
		x = self.layer3(x)
		nvtx.range_pop()

		nvtx.range_push("layer:conv5_x")
		x = self.layer4(x)
		nvtx.range_pop()

		x = self.avgpool(x)
		x = torch.flatten(x, 1)

		nvtx.range_push("layer:FC")
		x = self.fc(x)
		nvtx.range_pop()

		return x


def resnet50():
	return ResNet(Bottleneck, [3, 4, 6, 3])

#Create model
net = resnet50().cuda().half()
net.train()

#Create optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)

#Create synthetic input and label
x = torch.rand(32, 3, 224, 224).cuda().half()
target = torch.empty(32, dtype=torch.long).random_(1000).cuda()

with torch.autograd.profiler.emit_nvtx():
	profiler.start()
	output = net(x)
	loss = criterion(output, target)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	profiler.stop()
