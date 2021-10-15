import torch
from bottleneck import Bottleneck
torch.manual_seed(23337)

# use True to print layerwise sum for all outputs in reference code path
DEBUG = False#True

for stride, o_channel in [(1,32), (1,128), (2,32)]:
    print("testing stride ==", stride, ", in_channel == 32 , out_channel ==", o_channel)
    a_ = torch.randn(17,32,28,28)

    a = a_.cuda().half().to(memory_format=torch.channels_last).requires_grad_()
    model = Bottleneck(32,8,o_channel,stride=stride).cuda().half().to(memory_format=torch.channels_last)

    # test model
    b = model(a)
    b.mean().backward()
    d_grad = a.grad.float()
    a.grad = None
    torch.cuda.synchronize()

    if DEBUG:
        print("[DEBUG] ref dx :", d_grad.sum().item())
        # print wgrad. we don't need to reset since later cpp print before accumulation
        for i, w in enumerate(model.w_conv):
            print("[DEBUG] ref wgrad{} :".format(i+1), w.grad.sum().item())

    wgrads = []
    for w in model.w_conv:
        wgrads.append(w.grad.float())

    model.use_cudnn = True
    model.zero_grad()
    c = model(a)
    c.mean().backward()

    torch.cuda.synchronize()
    print("comparing native and channels_last:")
    print("max error fprop:", (b-c).abs().max().item(), "max elem:", b.abs().max().item())
    print("max error dgrad:", (d_grad-a.grad.float()).abs().max().item(), "max elem:", d_grad.abs().max().item())
    for i, (w, wgrad) in enumerate(zip(model.w_conv, wgrads)):
        print("max error wgrad{}:".format(i+1), (wgrad - w.grad.float()).abs().max().item(), "max elem:", wgrad.abs().max().item())

    nhwc_a = a_.permute(0,2,3,1).contiguous().cuda().half().requires_grad_()
    nhwc_model = Bottleneck(32,8,o_channel,stride=stride,explicit_nhwc=True, use_cudnn=True).cuda().half()
    for p,q in zip(model.parameters(), nhwc_model.parameters()):
        # model's storage is already in nhwc, we clone and assign to explicit nhwc model
        q.data.copy_(p.data.permute(0,2,3,1).contiguous())
    for p,q in zip(model.buffers(), nhwc_model.buffers()):
        q.data.copy_(p.data)

    d = nhwc_model(nhwc_a)
    d.mean().backward()
    torch.cuda.synchronize()

    # reset reference to cudnn channels_last permute
    #c_s = c.storage().tolist()
    #d_s = d.storage().tolist()
    #print(max([x-y for x,y in zip(c_s,d_s)]))
    c = c.contiguous(memory_format=torch.contiguous_format).permute(0,2,3,1).contiguous()
    d_grad = a.grad.float().permute(0,2,3,1).contiguous()
    wgrads = []
    for w in model.w_conv:
        wgrads.append(w.grad.float().permute(0,2,3,1).contiguous())

    torch.cuda.synchronize()
    print("comparing nhwc and channels_last:")
    print("max error fprop:", (d-c).abs().max().item(), "max elem:", c.abs().max().item())
    print("max error dgrad:", (d_grad-nhwc_a.grad.float()).abs().max().item(), "max elem:", d_grad.abs().max().item())
    for i, (w, wgrad) in enumerate(zip(nhwc_model.w_conv, wgrads)):
        print("max error wgrad{}:".format(i+1), (wgrad - w.grad.float()).abs().max().item(), "max elem:", wgrad.abs().max().item())
