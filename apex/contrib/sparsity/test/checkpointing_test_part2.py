from collections import OrderedDict

import torch
from apex.optimizers import FusedAdam
from apex.contrib.sparsity import ASP

def build_model(args):
    od = OrderedDict()
    for i in range(args.num_layers):
        if i == 0:
            od['linear_layer_%d' % (i+1)] = torch.nn.Linear(args.input_features, args.hidden_features)
            od['layer_norm_%d' % (i+1)] = torch.nn.LayerNorm([args.batch_size, args.hidden_features])
        elif i == args.num_layers-1:
            od['linear_layer_%d' % (i+1)] = torch.nn.Linear(args.hidden_features, args.output_features)
            od['layer_norm_%d' % (i+1)] = torch.nn.LayerNorm([args.batch_size, args.output_features])
        else:
            od['linear_layer_%d' % (i+1)] = torch.nn.Linear(args.hidden_features, args.hidden_features)
            od['layer_norm_%d' % (i+1)] = torch.nn.LayerNorm([args.batch_size, args.hidden_features])
    return torch.nn.Sequential(od)

def train_step(args, model, optimizer, input_batch, target_batch, step):
    predicted_target = model(input_batch)
    loss = ((predicted_target-target_batch)**2).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    step = step + 1
    #print("Step %d :: loss=%e" % (step, loss.item()))
    return step

def train_loop(args, model, optimizer, step, num_steps):
    for i in range(num_steps):
        input_batch = torch.randn([args.batch_size, args.input_features]).cuda()
        target_batch = torch.randn([args.batch_size, args.output_features]).cuda()
        step = train_step(args, model, optimizer, input_batch, target_batch, step)
    return step

def main(step, args, model_state_dict, optimizer_state_dict):
    #
    # PART2
    #

    model = build_model(args).cuda()
    one_ll = next(model.children()).weight
    optimizer = FusedAdam(model.parameters())
    ASP.init_model_for_pruning(model, args.pattern, verbosity=args.verbosity, whitelist=args.whitelist, allow_recompute_mask=args.allow_recompute_mask)
    ASP.init_optimizer_for_pruning(optimizer)

    torch.manual_seed(args.seed2)
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    print("Model sparsity is %s" % ("enabled" if ASP.sparsity_is_enabled() else "disabled"))

    # train for a few steps with sparse weights
    print("SPARSE :: ",one_ll)
    step = train_loop(args, model, optimizer, step, args.num_sparse_steps_2)

if __name__ == '__main__':
    checkpoint = torch.load("part1.chkp")
    class Args:
        verbosity = checkpoint['verbosity']
        seed = 4873
        seed2 = checkpoint['seed2']
        pattern = checkpoint['pattern']
        whitelist = checkpoint['whitelist']
        allow_recompute_mask = checkpoint['allow_recompute_mask']
        batch_size = 32
        input_features = 8
        output_features = 8
        hidden_features = 32
        num_layers = 4
        num_dense_steps = 2000
        num_sparse_steps = 3000
        num_sparse_steps_2 = 1000
        checkpoint_path = "part1.chkp"
    args = Args()

    main(checkpoint['step'], args, checkpoint['model_state_dict'], checkpoint['optimizer_state_dict'])
