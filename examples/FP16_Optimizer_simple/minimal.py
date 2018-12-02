import torch
from apex.fp16_utils import FP16_Optimizer

torch.backends.cudnn.benchmark = True

N, D_in, D_out = 64, 1024, 16

x = torch.randn(N, D_in, device='cuda', dtype=torch.half)
y = torch.randn(N, D_out, device='cuda', dtype=torch.half)

model = torch.nn.Linear(D_in, D_out).cuda().half()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

### Construct FP16_Optimizer
### FP16_Optimizer will ingest and remember the original optimizer's param_groups.
###
### Construct with static loss scaling...
optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.0)
### ...or dynamic loss scaling
# optimizer = FP16_Optimizer(optimizer, 
#                            dynamic_loss_scale=True,
#                            dynamic_loss_args={'scale_factor' : 2})
### dynamic_loss_args is optional, for "power users," and unnecessary in most cases.

loss_fn = torch.nn.MSELoss()

for t in range(200):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred.float(), y.float())
    ### Change loss.backward() to:
    optimizer.backward(loss)
    ###
    optimizer.step()

print("final loss = ", loss)
