import torch
from apex.fp16_utils import FP16_Optimizer

torch.backends.cudnn.benchmark = True

N, D_in, D_out = 64, 1024, 16

x = torch.randn(N, D_in, device='cuda', dtype=torch.half)
y = torch.randn(N, D_out, device='cuda', dtype=torch.half)

model = torch.nn.Linear(D_in, D_out).cuda().half()

optimizer = torch.optim.LBFGS(model.parameters())
### Construct FP16_Optimizer
optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.0)
###

loss_fn = torch.nn.MSELoss()

for t in range(5):
    def closure():
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred.float(), y.float())
        ### Change loss.backward() within the closure to: ###
        optimizer.backward(loss)
        ###
        return loss
    loss = optimizer.step(closure)

print("final loss = ", loss)
