#!/usr/bin/env python3

"""
Example to run pyprof with imagenet models.
"""

import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.cuda.profiler as profiler
import argparse

from apex import pyprof
from apex.optimizers import FusedAdam

def parseArgs():
	parser = argparse.ArgumentParser(prog=sys.argv[0], description="Run popular imagenet models.")

	parser.add_argument("-m",
		type=str,
		default="resnet50",
		choices=["alexnet", "densenet121", "densenet161", "densenet169", "densenet201", "googlenet", "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3", "mobilenet_v2", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "squeezenet1_0", "squeezenet1_1", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn", "inception_v3"],
		help="Model.")

	parser.add_argument("-b",
		type=int,
		default=32,
		help="Batch size.")

	parser.add_argument("-o",
		type=str,
		default="adam",
		choices=["adam", "sgd"],
		help="Optimizer.")

	args = parser.parse_args()
	return args

d = {
	"alexnet":				{'H': 224, 'W': 224, 'opts': {}},

	"densenet121":			{'H': 224, 'W': 224, 'opts': {}},
	"densenet161":			{'H': 224, 'W': 224, 'opts': {}},
	"densenet169":			{'H': 224, 'W': 224, 'opts': {}},
	"densenet201":			{'H': 224, 'W': 224, 'opts': {}},

	"googlenet":			{'H': 224, 'W': 224, 'opts': {'aux_logits': False}},

	"mnasnet0_5":			{'H': 224, 'W': 224, 'opts': {}},
	"mnasnet0_75":			{'H': 224, 'W': 224, 'opts': {}},
	"mnasnet1_0":			{'H': 224, 'W': 224, 'opts': {}},
	"mnasnet1_3":			{'H': 224, 'W': 224, 'opts': {}},

	"mobilenet_v2":			{'H': 224, 'W': 224, 'opts': {}},

	"resnet18":				{'H': 224, 'W': 224, 'opts': {}},
	"resnet34":				{'H': 224, 'W': 224, 'opts': {}},
	"resnet50":				{'H': 224, 'W': 224, 'opts': {}},
	"resnet101":			{'H': 224, 'W': 224, 'opts': {}},
	"resnet152":			{'H': 224, 'W': 224, 'opts': {}},

	"resnext50_32x4d":		{'H': 224, 'W': 224, 'opts': {}},
	"resnext101_32x8d":		{'H': 224, 'W': 224, 'opts': {}},

	"wide_resnet50_2":		{'H': 224, 'W': 224, 'opts': {}},
	"wide_resnet101_2":		{'H': 224, 'W': 224, 'opts': {}},

	"shufflenet_v2_x0_5": 	{'H': 224, 'W': 224, 'opts': {}},
	"shufflenet_v2_x1_0": 	{'H': 224, 'W': 224, 'opts': {}},
	"shufflenet_v2_x1_5": 	{'H': 224, 'W': 224, 'opts': {}},
	"shufflenet_v2_x2_0":	{'H': 224, 'W': 224, 'opts': {}},

	"squeezenet1_0":		{'H': 224, 'W': 224, 'opts': {}},
	"squeezenet1_1":		{'H': 224, 'W': 224, 'opts': {}},

	"vgg11":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg11_bn":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg13":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg13_bn":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg16":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg16_bn":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg19":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg19_bn":				{'H': 224, 'W': 224, 'opts': {}},

	"inception_v3":			{'H': 299, 'W': 299, 'opts': {'aux_logits': False}},
	}

def main():
	args = parseArgs()

	pyprof.nvtx.init()
#	pyprof.nvtx.wrap(fused_adam_cuda, 'adam')

	N = args.b
	C = 3
	H = d[args.m]['H']
	W = d[args.m]['W']
	opts = d[args.m]['opts']
	classes = 1000

	net = getattr(models, args.m)
	net = net(**opts).cuda().half()
	net.train()

	x = torch.rand(N, C, H, W).cuda().half()
	target = torch.empty(N, dtype=torch.long).random_(classes).cuda()

	criterion = nn.CrossEntropyLoss().cuda()
	if (args.o == "sgd"):
		optimizer = torch.optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
	elif (args.o == "adam"):
		optimizer = FusedAdam(net.parameters())
	else:
		assert False

	#Warm up without profiler
	for i in range(2):
		output = net(x)
		loss = criterion(output, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	with torch.autograd.profiler.emit_nvtx():
		profiler.start()
		output = net(x)
		loss = criterion(output, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		profiler.stop()

if __name__ == "__main__":
	main()
