# Word-level language modeling RNN

This example is based on [https://github.com/pytorch/examples/tree/master/word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model).
It trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

`main.py` with the `--fp16` argument demonstrates mixed precision training with manual management of master parameters and loss scaling.

`main_fp16_optimizer.py` with `--fp16` demonstrates use of `apex.fp16_utils.FP16_Optimizer` to automatically manage master parameters and loss scaling.

```bash
python main.py --cuda --epochs 6        # Train a LSTM on Wikitext-2 with CUDA, reaching perplexity of 117.61
python main.py --cuda --epochs 6 --tied # Train a tied LSTM on Wikitext-2 with CUDA, reaching perplexity of 110.44
python main.py --cuda --tied            # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs, reaching perplexity of 87.17
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

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40           # Test perplexity of 80.97
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied    # Test perplexity of 75.96
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40        # Test perplexity of 77.42
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied # Test perplexity of 72.30
```

Perplexities on PTB are equal or better than
[Recurrent Neural Network Regularization (Zaremba et al. 2014)](https://arxiv.org/pdf/1409.2329.pdf)
and are similar to [Using the Output Embedding to Improve Language Models (Press & Wolf 2016](https://arxiv.org/abs/1608.05859) and [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling (Inan et al. 2016)](https://arxiv.org/pdf/1611.01462.pdf), though both of these papers have improved perplexities by using a form of recurrent dropout [(variational dropout)](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks).
