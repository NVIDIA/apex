# ChannelPermutations

Standalone code to reproduce results in "[Channel Permutations for N:M Sparsity](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html)," Jeff Pool and Chong Yu, NeurIPS 2021.

Three search strategies are supported: randomly generating permutations and checking quality, greedily swapping columns until convergence (i.e. TETRIS adapted for 2:4 sparsity), and the technique presented in the above paper, optimizing stripe groups.  This tool will apply these strategies, as configured below, to either a randomly-generated matrix or an .npy file (typically from a real network) and report the efficacy and runtime of the strategy.

## Quick Start

### Installation

#### GPU path

Requirements:
- CUDA
- pybind11

A container such as `nvcr.io/nvidia/pytorch:21.12-py3` satisfies these requirements.

Installation (from this directory):
```
pushd ../permutation_search_kernels/CUDA_kernels
nvcc -O3 -shared -Xcompiler -fPIC -Xcompiler -DTORCH_EXTENSION_NAME=permutation_search_cuda -std=c++11 $(python3 -m pybind11 --includes) permutation_search_kernels.cu -o ../permutation_search_cuda$(python3-config --extension-suffix)
popd
```

#### CPU path

Only NumPy is required for CPU-only execution.

### Important arguments

`python3 permutation_test.py` will tell you all the available arguments and alert you about required arguments:
```
    usage: permutation_test.py [-h] [--infile INFILE] [--channels CHANNELS] [--filters FILTERS] 
                               [--verbosity VERBOSITY] [--seed SEED] [--pretty_print PRETTY_PRINT] 
                               [--unstructured UNSTRUCTURED] [--gpu GPU] [--check_permutation CHECK_PERMUTATION] 
                               [--intermediate_steps INTERMEDIATE_STEPS] [--print_permutation PRINT_PERMUTATION]
                               strategy [strategy ...]
    permutation_test.py: error: the following arguments are required: strategy
```

Detailed information about each argument:

- `--infile` (string) accepts .npy files with weights dumped from some model checkpoint.  By default, the input file is `'random'`, which will generate a random 2D matrix with `CHANNELS` columns and `FILTERS` rows.
- `--channels` and `--filters` (unsigned integers) specify the size of the randomly-generated matrix if there is no input file specified.
- `--verbosity` (unsigned integer) controls the amount of debug and status information printed.  `0` is just the important data, `11` can give periodic status details, and higher integers provide increasingly more detail.
- `--seed` (unsigned integer) allows for changing the random seed, which will affect the random matrix generation, random permutations generated, and columns swapped for bounded regressions.
- `--pretty_print` (bool) prints a pretty graph by default (below), but disabling will generate output friendly for redirecting to a .csv file.
- `--unstructured` (float) will apply unstructured pruning to the matrix before searching for permutations.  A negative value will find the minimum unstructured sparsity for which a search strategy can find a perfect permutation and not create any extra zeros.
- `--gpu` (bool) uses CUDA kernels by default (if they are built and there is a GPU available), but you can override this to run on the CPU.
- `--check_permutation` (bool) makes sure the permutation tracked during the search process matches the one that's recovered directly from the permuted matrix.
- `--intermediate_steps` (unsigned integer) will emit permutations with efficacies equally dividing the distance between the default order and the best permutation found.
- `--print_permutation` (bool) prints the permutation found for each strategy.

Finally, after these optional arguments, provide the search strategies desired.  There are three strategies offered:
- `random,<num_seeds=10>`
- `channel_swaps,<bounded_regressions=100>`
- `optimize_stripe_groups,<stripe_group_size_in_columns=8>,<bounded_regressions=100>`

### Launch a test with interesting search strategies

Now that kernels are built, you can use them to accelerate the search, which can be quite time-consuming without using the GPU.  Below, we report results on a number of interesting strategies for a 64-column, 128-row random matrix using a V100 accelerator.

    $ python3 permutation_test.py --channels 64 --filters 128 channel_swap,0 channel_swap,100 channel_swap,1000 optimize_stripe_groups,8,0 optimize_stripe_groups,8,100 optimize_stripe_groups,8,1000 optimize_stripe_groups,12,0 random,1000 random,10000 random,100000
    Found permutation search CUDA kernels for standalone testing
    Found 2 gpus
    strategy                           ,      magnitude,       efficacy,       duration
    unpruned                           ,       4083.169,       -       ,       -       
    unstructured                       ,       3060.238,       -       ,       -       
    50% rows                           ,       3042.332,          100.0,       -       
    default 2:4                        ,       2852.376,            0.0,          0.000
    channel_swap,0                     ,       2913.352,           32.1,          0.214               
    channel_swap,100                   ,       2914.174,           32.5,          2.249               
    channel_swap,1000                  ,       2920.694,           36.0,         20.248               
    optimize_stripe_groups,8,0         ,       2919.757,           35.5,          0.013               
    optimize_stripe_groups,8,100       ,       2919.758,           35.5,          0.152               
    optimize_stripe_groups,8,1000      ,       2919.935,           35.6,          1.387               
    optimize_stripe_groups,12,0        ,       2921.947,           36.6,          0.860               
    random,1000                        ,       2873.380,           11.1,          0.116               
    random,10000                       ,       2873.603,           11.2,          1.149               
    random,100000                      ,       2879.129,           14.1,         11.510   

For this particular input, the `channel_swap` strategy requires 1000 bounded regressions in order to surpass the efficacy of optimizing two stripe groups (8 columns) without any bounded regressions, but allowing 1000 bounded regressions when optimizing two stripe groups is slightly worse than swapping channels with 1000 bounded regressions.  Optimizing *three* stripe groups at a time outperforms all the other approaches by a wide margin.  Testing many random permutations is inefficient and ineffective.

Without GPU acceleration, these tests would be much slower (though they find the same final permutations):

    $ python3 permutation_test.py --gpu 0 --channels 64 --filters 128 channel_swap,0 channel_swap,100 optimize_stripe_groups,8,0 optimize_stripe_groups,8,100 random,1000
    strategy                           ,      magnitude,       efficacy,       duration
    unpruned                           ,       4083.169,       -       ,       -       
    unstructured                       ,       3060.238,       -       ,       -       
    50% rows                           ,       3042.332,          100.0,       -       
    default 2:4                        ,       2852.377,            0.0,          0.016
    channel_swap,0                     ,       2913.351,           32.1,         55.972
    channel_swap,100                   ,       2914.174,           32.5,        450.025
    optimize_stripe_groups,8,0         ,       2919.759,           35.5,         60.653
    optimize_stripe_groups,8,100       ,       2919.759,           35.5,        465.709
    random,1000                        ,       2873.381,           11.1,         14.889


### Perform the ablation study from Table 1

`bash ablation_studies.sh` will generate the results for the ablation study, showing the relative importance of the bounded regressions and stripe group greedy phase.

### Generate the runtime results from Table 3

`bash runtime_table.sh` will generate the search strategies' efficacies and runtime shown in Table 3.

### Traverse permutation space (as in Figure 3)

We developed a heuristic approach to interpolating between permutations which allows us to find permutations with efficacies that evenly divide some range.  The `--intermediate_steps <N>` argument can be used to emit such a sequence of permutations:

    $ python3 permutation_test.py --channels 64 --filters 128 --intermediate_steps 7 --print_permutation 1 optimize_stripe_groups,8,0
    Found permutation search CUDA kernels for standalone testing
    Found 2 gpus
    strategy                           ,      magnitude,       efficacy,       duration
    unpruned                           ,       4083.169,       -       ,       -
    unstructured                       ,       3060.238,       -       ,       -
    50% rows                           ,       3042.332,          100.0,       -
    default 2:4                        ,       2852.377,            0.0,          0.000
    (2859.8855, [2, 8, 14, 24, 9, 12, 13, 15, 4, 5, 6, 7, 0, 1, 3, 46, 40, 41, 42, 43, 32, 33, 34, 35, 25, 26, 27, 55, 16, 17, 18, 58, 20, 21, 22, 23, 38, 60, 61, 63, 11, 44, 45, 47, 36, 37, 39, 62, 10, 28, 29, 30, 31, 52, 53, 54, 19, 56, 57, 59, 48, 49, 50, 51])
    (2870.1387, [5, 6, 7, 41, 9, 12, 13, 35, 0, 1, 3, 46, 30, 40, 42, 43, 2, 32, 33, 34, 25, 26, 27, 55, 16, 17, 18, 58, 20, 21, 22, 23, 38, 60, 61, 63, 11, 44, 45, 47, 36, 37, 39, 62, 4, 10, 28, 29, 31, 52, 53, 54, 19, 56, 57, 59, 15, 48, 49, 50, 8, 14, 24, 51])
    (2878.0679, [36, 37, 39, 62, 9, 12, 13, 35, 0, 3, 16, 46, 30, 40, 42, 43, 2, 5, 32, 33, 23, 26, 27, 55, 1, 20, 21, 22, 38, 60, 61, 63, 11, 44, 45, 47, 6, 7, 25, 41, 4, 10, 28, 29, 31, 52, 53, 54, 19, 56, 57, 59, 15, 48, 49, 50, 8, 14, 24, 51, 17, 18, 34, 58])
    (2884.8323, [9, 12, 35, 54, 0, 3, 16, 46, 30, 40, 42, 43, 2, 5, 32, 33, 23, 26, 27, 55, 11, 44, 45, 47, 36, 37, 39, 62, 4, 10, 28, 29, 31, 52, 53, 60, 19, 21, 56, 57, 15, 48, 49, 50, 8, 14, 24, 51, 17, 18, 34, 58, 6, 7, 25, 41, 1, 13, 20, 22, 38, 59, 61, 63])
    (2894.9697, [9, 12, 33, 35, 0, 3, 16, 46, 2, 5, 32, 52, 23, 26, 27, 55, 11, 44, 45, 47, 36, 37, 39, 62, 4, 10, 28, 29, 19, 21, 50, 56, 15, 43, 48, 49, 8, 14, 24, 51, 17, 18, 34, 58, 6, 7, 25, 41, 1, 13, 20, 22, 38, 59, 61, 63, 30, 40, 42, 54, 31, 53, 57, 60])
    (2901.5115, [9, 12, 35, 56, 0, 3, 16, 46, 23, 26, 27, 55, 33, 36, 37, 39, 4, 10, 28, 29, 19, 21, 45, 50, 8, 14, 24, 51, 17, 18, 34, 58, 6, 7, 25, 41, 1, 13, 20, 22, 38, 59, 61, 63, 30, 40, 42, 54, 31, 53, 57, 60, 2, 5, 32, 52, 15, 43, 49, 62, 11, 44, 47, 48])
    (2910.2043, [4, 10, 28, 37, 9, 12, 35, 56, 0, 3, 16, 46, 23, 33, 36, 39, 8, 14, 24, 51, 17, 18, 34, 58, 6, 7, 25, 41, 1, 13, 20, 22, 38, 59, 61, 63, 30, 40, 42, 54, 31, 53, 57, 60, 2, 5, 32, 52, 15, 43, 49, 62, 11, 44, 47, 48, 19, 21, 45, 50, 26, 27, 29, 55])
    optimize_stripe_groups,8,0         ,       2919.757,           35.5,          0.015
    [0, 9, 12, 35, 4, 10, 28, 37, 50, 19, 45, 21, 34, 17, 18, 58, 16, 46, 39, 3, 49, 43, 15, 62, 6, 7, 41, 25, 48, 11, 44, 47, 13, 20, 22, 1, 55, 29, 26, 27, 5, 2, 32, 52, 40, 30, 42, 54, 53, 57, 60, 31, 36, 56, 23, 33, 59, 38, 61, 63, 51, 24, 14, 8]

### Transform unstructured sparsity to structured sparsity (as in Figure 4)

If you have a directory with .npy weight files for each layer of a network, `bash unstructured_study.sh <path_to_directory> <network_name>` will perform a binary search for each file to find the minimum unstructured sparsity required to transparently transform that layer with a number of permutation search techniques; this file was used to generate Figure 4, using weights dumped from a pre-trained ResNet50 in Torchvision.

## References

The baseline algorithm which we adapated for use with 2:4 sparsity and upon which we improved is "[TETRIS](https://papers.nips.cc/paper/2018/hash/89885ff2c83a10305ee08bd507c1049c-Abstract.html): TilE-matching the TRemendous Irregular Sparsity," Ji et al., NeurIPS 2018.

If you want to use this technique when generating a 2:4 sparse network for inference, we've packaged it into our [ASP](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity) library - this will perform the permutation searches for each layer as required, as well as fix up neighboring layers so there are no extra operations inserted at runtime.

## Citation

If you use this idea or code in your own research, please cite the [paper](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html) that describes it:

```
@inproceedings{pool2021channel,
  author    = {Pool, Jeff and Yu, Chong},
  booktitle = {Advances in Neural Information Processing Systems ({NeurIPS})},
  title     = {Channel Permutations for {N:M} Sparsity},
  url       = {https://proceedings.neurips.cc/paper/2021/file/6e8404c3b93a9527c8db241a1846599a-Paper.pdf},
  volume    = {34},
  year      = {2021}
}

```

