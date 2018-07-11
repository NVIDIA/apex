import math
import torch

from . import utils
from apex_C import scale_check_overflow

def run(handle, model, loss_fn):
    print('Running amp debug.\n')
    # 1) Check for overflows w/o loss scaling in fp16
    print('Checking for overflow without loss scale...')
    _reset(model)
    hooks = add_print_overflow_hooks(model)
    loss_fn().backward()
    for h in hooks:
        h.remove()
    print('Done.')
    print('\nGrad stats\n' + ('=' * 10))
    for name, p in model.named_parameters():
        if p.grad is not None:
            amax = torch.max(torch.abs(p.grad.data))
            norm = torch.norm(p.grad.data)
            print('{}:\n  Amax: {}\n  Norm: {}'.format(name, amax, norm))

    # 2) Find largest non-overflow loss-scale
    max_loss_scale = 0.
    for loss_scale in [2.**k for k in reversed(range(25))]:
        _reset(model)
        (loss_fn() * loss_scale).backward()
        if not any_overflows(model.parameters()):
            max_loss_scale = loss_scale
            break

    print('\nGrad comparison\n' + ('=' * 15))
    if max_loss_scale == 0.:
        print('Overflow at loss scale == 1. Look above to debug.')
        return
    print('Maximum loss scale is {} (2**{}).'.format(max_loss_scale,
                                                       math.log2(max_loss_scale)))

    # 3) Compare grads computed in fp32 and fp16
    _reset(model)
    with handle._disable_casts():
        loss_fn().backward()
    fp32_grads = []
    for p in model.parameters():
        if p.grad is not None:
            fp32_grads.append(p.grad.data.detach().clone())
        else:
            fp32_grads.append(None)

    _reset(model)
    (loss_fn() * max_loss_scale).backward()
    for fp32_grad, (name, p) in zip(fp32_grads, model.named_parameters()):
        if fp32_grad is None:
            continue
        fp16_grad = p.grad.data * (1. / max_loss_scale)
        diff = torch.max(torch.abs(fp32_grad - fp16_grad))
        sim = cosine_sim(fp32_grad, fp16_grad)
        print('{}:\n  max_diff: {}\n  cosine_sim: {}'.format(name, diff, sim))

# NB: this doesn't reset cuDNN RNN dropout state, since it persists across calls
# TODO(carl): is there a way around that?
def _reset(model):
    model.zero_grad()
    torch.manual_seed(0)

def log_overflow_hook(module_name, hook_input_names):
    def hook(module, *args):
        print_name = '{} ({})'.format(module_name, type(module))
        for name, arg_lst in zip(hook_input_names,
                                 [utils.as_iterable(x) for x in args]):
            for i, arg in enumerate(arg_lst):
                if arg is None or not utils.is_fp_tensor(arg):
                    continue
                if utils.is_nested(arg):
                    xs = arg
                else:
                    xs = [arg]
                for x in xs:
                    if torch.isnan(x).any():
                        print('NaN in {} - {}[{}]'.format(print_name, name, i))
                    if (x.abs() == float('inf')).any():
                        print('Inf in {} - {}[{}]'.format(print_name, name, i))
    return hook

def add_print_overflow_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(
            log_overflow_hook(name, ['forward-input', 'forward-output'])))
        hooks.append(module.register_backward_hook(
            log_overflow_hook(name, ['grad-input', 'grad-output'])))
    return hooks

def any_overflows(parameters):
    buf = torch.cuda.ByteTensor(1024,).zero_()
    for p in parameters:
        if p.grad is not None:
            scale_check_overflow(p.grad.data,
                                 1.,
                                 buf)
    return bool(buf.any())

def cosine_sim(a, b):
    dot = torch.dot(a.view(-1), b.view(-1))
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    return dot / (norm_a * norm_b)
