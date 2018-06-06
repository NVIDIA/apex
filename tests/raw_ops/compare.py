import torch
import numpy as np

def compare(cuda_out, pt_out, pt_out_control, rows):

    # print(                    "Pytorch ops in fp16:  ", pt_out        )
    # print(                          "Kernel result:  ", cuda_out      )
    # print("Control (Pytorch ops, sticking to fp32):  ", pt_out_control)
    
    # Make upconverted copies for error check against fp32 control
    cuda_out_fp32 = cuda_out.float()
    pt_out_fp32 = pt_out.float()
    
    # Flatten all but the slowest dimension
    cuda_out       =       cuda_out.view(rows,-1)
    pt_out         =         pt_out.view(rows,-1)
    cuda_out_fp32  =  cuda_out_fp32.view(rows,-1)
    pt_out_fp32    =    pt_out_fp32.view(rows,-1)
    pt_out_control = pt_out_control.view(rows,-1)
   
    cuda_maxdiffs, cuda_maxdiff_locs = torch.max((pt_out_control - cuda_out_fp32).abs(),1)
    pt_maxdiffs, pt_maxdiff_locs     = torch.max((pt_out_control - pt_out_fp32  ).abs(),1)
    
    print(    "cuda_maxdiffs = ", cuda_maxdiffs    )
    # print("cuda_maxdiff_locs = ", cuda_maxdiff_locs)
    print(      "pt_maxdiffs = ", pt_maxdiffs      )
    # print(  "pt_maxdiff_locs = ", pt_maxdiff_locs  )
    
    row_indices = torch.LongTensor(np.arange(rows))
    
    # print("cuda_out at cuda_maxdiff_locs in each row:")
    # # bizarrely, this will work if you do it at the python prompt:
    # # print(cuda_out[row_indices,cuda_maxdiff_locs])
    # # ...but it only seems to work here if you wrap with numpy arrays:
    # print(      cuda_out[np.array(row_indices),np.array(cuda_maxdiff_locs)])
    # print("pt_out_control at cuda_maxdiff_locs in each row:")
    # print(pt_out_control[np.array(row_indices),np.array(cuda_maxdiff_locs)])
    # 
    # print("pt_out at pt_maxdiff_locs in each row:"          )
    # print(        pt_out[np.array(row_indices),np.array(pt_maxdiff_locs)])
    # print("pt_out_control at pt_maxdiff_locs in each row:"  )
    # print(pt_out_control[np.array(row_indices),np.array(pt_maxdiff_locs)])
