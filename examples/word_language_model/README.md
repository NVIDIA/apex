# Word-level language modeling RNN

This example is based on [https://github.com/pytorch/examples/tree/master/word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model).
It trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

`main.py` with the `--fp16` argument demonstrates mixed precision training with manual management of master parameters and loss scaling.

`main_fp16_optimizer.py` with `--fp16` demonstrates use of `apex.fp16_utils.FP16_Optimizer` to automatically manage master parameters and loss scaling.

These examples are intended as an illustration of the mixed precision recipe, not necessarily as a performance showcase.  However, they do demonstrate certain best practices.

First, a default loss scale of 128.0 is used.  In our testing, this improves converged test perplexity modestly with mixed precision, from around 93 with loss scale 1.0 to around 90 with loss scale 128.0.

Second, to enable Tensor Core use with `--fp16` and improve performance, dimensions that participate in GEMMs in the model are made multiples of 8.  Specifically, these are
* dictionary length (ntokens in `main.py`),
* embedding size (`--emsize`),
* hidden size (`--nhid`), and
* batch size (`--batch_size`).

The dictionary length is a property of the dataset, and is not controlled by a command line argument. In `main.py`, `corpus = data.Corpus(args.data, pad_to_multiple_of=8)` and the `Corpus` constructor in
`data.py` ensure that the dictionary length is a multiple of 8.

Also, for mixed precision performance, a good general rule is: the more work you give the GPU, the better.  Bigger models and larger batch sizes supply the cores with more work and do a better job saturating the device.  A (very rough) way to check if you're saturating the device is to run nvidia-smi from another terminal, and see what fraction of device memory you're using.  This will tell you how much leeway you have to increase model or batch size.

```bash
python main.py --cuda --epochs 6        # Train a LSTM on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --fp16 # Train a LSTM on Wikitext-2 with CUDA and mixed precision
python main.py --cuda --epochs 6 --tied # Train a tied LSTM on Wikitext-2 with CUDA
python main.py --cuda --tied            # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs
python generate.py                      # Generate samples from the trained LSTM model.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

## Usage for `main.py` and `main_fp16_optimizer.py`

```bash
usage: main.py [-h] [--data DATA] [--model MODEL] [--emsize EMSIZE]
               [--nhid NHID] [--nlayers NLAYERS] [--lr LR] [--clip CLIP]
               [--epochs EPOCHS] [--batch_size N] [--bptt BPTT]
               [--dropout DROPOUT] [--tied] [--seed SEED] [--cuda]
               [--log-interval N] [--save SAVE] [--fp16]
               [--static-loss-scale STATIC_LOSS_SCALE]

PyTorch Wikitext-2 RNN/LSTM Language Model

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --fp16                Run model in pseudo-fp16 mode (fp16 storage fp32
                        math).
  --static-loss-scale STATIC_LOSS_SCALE
                        Static loss scale, positive power of 2 values can
                        improve fp16 convergence.

```

`main_fp16_optimizer` also accepts the optional flag
```bash
  --dynamic-loss-scale  Use dynamic loss scaling. If supplied, this argument
                        supersedes --static-loss-scale.
```
which triggers the use of dynamic loss scaling.  Supplying `--dynamic-loss-scale` will override the `--loss_scale` argument, if any.

With these arguments, a variety of models can be tested.  For example

```bash
python main.py --cuda --emsize 656 --nhid 656 --dropout 0.5 --epochs 40
python main.py --cuda --emsize 656 --nhid 656 --dropout 0.5 --epochs 40 --tied
python main.py --cuda --emsize 1504 --nhid 1504 --dropout 0.65 --epochs 40
python main.py --cuda --emsize 1504 --nhid 1504 --dropout 0.65 --epochs 40 --tied
```
