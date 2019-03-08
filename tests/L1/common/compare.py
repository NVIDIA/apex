import argparse
import torch

parser = argparse.ArgumentParser(description='Compare')
parser.add_argument('--opt-level', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--fused-adam', action='store_true')
args = parser.parse_args()

base_file = str(args.opt_level) + "_" +\
            str(args.loss_scale) + "_" +\
            str(args.keep_batchnorm_fp32) + "_" +\
            str(args.fused_adam)

file_e = "True_" + base_file
file_p = "False_" + base_file

dict_e = torch.load(file_e)
dict_p = torch.load(file_p)

torch.set_printoptions(precision=10)

print(file_e)
print(file_p)

for n, (i_e, i_p) in enumerate(zip(dict_e["Iteration"], dict_p["Iteration"])):
    assert i_e == i_p, "i_e = {}, i_p = {}".format(i_e, i_p)

    loss_e = dict_e["Loss"][n]
    loss_p = dict_p["Loss"][n]
    assert loss_e == loss_p, "Iteration {}, loss_e = {}, loss_p = {}".format(i_e, loss_e, loss_p)
    print("{:4} {:15.10f} {:15.10f} {:15.10f} {:15.10f}".format(
          i_e,
          loss_e,
          loss_p,
          dict_e["Speed"][n],
          dict_p["Speed"][n]))
