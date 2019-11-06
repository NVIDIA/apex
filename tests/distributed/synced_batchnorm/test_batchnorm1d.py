import torch
import apex

model = apex.parallel.SyncBatchNorm(4).cuda()
model.weight.data.uniform_()
model.bias.data.uniform_()
data = torch.rand((8,4)).cuda()

model_ref = torch.nn.BatchNorm1d(4).cuda()
model_ref.load_state_dict(model.state_dict())
data_ref = data.clone()

output = model(data)
output_ref = model_ref(data_ref)

assert(output.allclose(output_ref))
assert(model.running_mean.allclose(model_ref.running_mean))
assert(model.running_var.allclose(model_ref.running_var))
