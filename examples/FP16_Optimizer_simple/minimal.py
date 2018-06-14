import torch
from torch.autograd import Variable
from apex.fp16_utils import FP16_Optimizer

torch.backends.cudnn.benchmark = True

N, D_in, D_out = 64, 1024, 16

x = Variable(torch.cuda.FloatTensor(N, D_in ).normal_()).half()
y = Variable(torch.cuda.FloatTensor(N, D_out).normal_()).half()

model = torch.nn.Linear(D_in, D_out).cuda().half()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
### Construct FP16_Optimizer with static loss scaling ###
optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.0)
### ...or construct with dynamic loss scaling ###
# optimizer = FP16_Optimizer(optimizer, 
#                            dynamic_loss_scale=True,
#                            dynamic_loss_args={'scale_factor' : 4})
### dynamic_loss_args is optional, for "power users,"  and unnecessary in most cases.

loss_fn = torch.nn.MSELoss()

for t in range(1000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred.float(), y.float())
    ### Change loss.backward() to: ###
    optimizer.backward(loss)
    ###
    optimizer.step()

print("final loss = ", loss)
