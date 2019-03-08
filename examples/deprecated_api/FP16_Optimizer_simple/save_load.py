import torch
from apex.fp16_utils import FP16_Optimizer

torch.backends.cudnn.benchmark = True

N, D_in, D_out = 64, 1024, 16

x = torch.randn(N, D_in, device='cuda', dtype=torch.half)
y = torch.randn(N, D_out, device='cuda', dtype=torch.half)

model = torch.nn.Linear(D_in, D_out).cuda().half()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
### Construct FP16_Optimizer with static loss scaling...
optimizer = FP16_Optimizer(optimizer, static_loss_scale=128.0)
### ...or dynamic loss scaling
# optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

loss_fn = torch.nn.MSELoss()

# The checkpointing shown here is identical to what you'd use without FP16_Optimizer.
#
# We save/load checkpoints within local scopes, so the "checkpoint" object
# does not persist.  This helps avoid dangling references to intermediate deserialized data,
# and is good practice for Pytorch in general, not just with FP16_Optimizer.
def save_checkpoint():
    checkpoint = {}
    checkpoint['model'] = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    torch.save(checkpoint, 'saved.pth')

def load_checkpoint():
    checkpoint = torch.load('saved.pth', 
        map_location = lambda storage, loc: storage.cuda(torch.cuda.current_device()))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

for t in range(100):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred.float(), y.float())
    optimizer.backward(loss) ### formerly loss.backward()
    optimizer.step()

save_checkpoint()

load_checkpoint()

for t in range(100):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred.float(), y.float())
    optimizer.backward(loss) ### formerly loss.backward()
    optimizer.step()

print("final loss = ", loss)
