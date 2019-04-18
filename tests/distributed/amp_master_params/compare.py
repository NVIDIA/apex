import torch

model_params_rank0 = torch.load("rank0model.pth",
                           map_location = lambda storage, loc: storage.cuda(0))
model_params_rank1 = torch.load("rank1model.pth",
                                 map_location = lambda storage, loc: storage.cuda(0))
master_params_rank0 = torch.load("rank0master.pth",
                                 map_location = lambda storage, loc: storage.cuda(0))
master_params_rank1 = torch.load("rank1master.pth",
                                 map_location = lambda storage, loc: storage.cuda(0))

for model_rank0, model_rank1, master_rank0, master_rank1 in zip(
        model_params_rank0,
        model_params_rank1,
        master_params_rank0,
        master_params_rank1):
    assert torch.allclose(model_rank0, model_rank1), "Model param mismatch"
    assert torch.allclose(master_rank0, master_rank1), "Master param mismatch"
    # Some debugging/investigation assistance code:
    # maxval, maxind = torch.max(((torch.abs(model_rank0).float())/torch.abs(master_rank0)).view(-1), 0)
    # offending_val_half = model_rank0.view(-1)[maxind.item()]
    # offending_val_float = master_rank0.view(-1)[maxind.item()]
    # print(maxval.item(), maxind.item(), offending_val_half.item(), offending_val_float.item(),
    #       offending_val_float.half().item())
    # rtol needs to be > 2^-11 because of denormals...
    assert torch.allclose(model_rank0, master_rank0.half(), rtol=.005), "Model-master mismatch"

print("OK:  Model and master params match across ranks.")
