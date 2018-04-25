import torch
from torch.autograd import Variable
# import apex
import numpy as np

torch.manual_seed(2)
torch.cuda.manual_seed(2)
# torch.cuda.manual_seed_all(2)
torch.set_printoptions(precision=10)

rows = 3
cols = 20
dims = rows, cols

# Incoming gradient vectors we will use later
# Need to create the fp16 versions as a half() copy of a Tensor first rather than
# a Variable, because if you create pt_input_control as a Variable then say
# pt_input_fp16 = pt_input_control.half(), you are accidentally making pt_input_fp16 part of 
# pLpOutput_control's computational graph, so it will not be a leaf!
pt_input_control = Variable(torch.randn(*dims).cuda(), requires_grad=True)
# pt_input_control = torch.ones(*dims).cuda()
pt_input_fp16    = pt_input_control.half()

pt_output_fp16    = pt_input_fp16.sum()
pt_output_control = pt_input_control.sum()
print("After sum()s, before backwards:")
print("pt_output_control.requires_grad = ", pt_output_control.requires_grad)
print("pt_output_control.volatile = ", pt_output_control.volatile)
print("pt_input_control.grad = ", pt_input_control.grad)
print("pt_input_fp16.grad = ", pt_input_fp16.grad)
print("\n\n")

pt_output_fp16.backward() # pt_input_fp16 is not the leaf of this graph, pt_input_control is.
print("After pt_output_fp16.backward():")
print("pt_input_control.grad = ", pt_input_control.grad)
print("pt_input_fp16.grad = ", pt_input_fp16.grad)
print("\n\n")
pt_output_control.backward() # Both backward() calls have pt_input_control as leaves, and so
                             # will accumulate gradients into pt_input_control.grad
print("After pt_output_control.backward():")
print("pt_input_control.grad = ", pt_input_control.grad)
print("pt_input_fp16.grad = ", pt_input_fp16.grad)
print("\n\n")
print("pt_output_control = ", pt_output_control)
print("pt_output_fp16 = ", pt_output_fp16)


