# amp: Automatic Mixed Precision

amp is an experimental tool to enable mixed precision training in
PyTorch with _extreme_ simplicity and overall numerical safety. It
does so by employing a whitelist / blacklist model:
- Any function on the whitelist casts its input arguments to
  fp16. These are functions like `torch.conv2d` that can take
  advantage of TensorCore execution.
- Any function on the blacklist casts its input arguments to
  fp32. These are functions like `torch.exp` or loss functions that
  have trouble with the numerical properties of fp16.
- Any other function passes along its input types to its outputs. Care
  is taken so that multi-argument functions or methods
  (e.g. `torch.tensor.__add__`) can handle mixed type inputs. They
  simply promote all inputs to have the widest type of any input.

The PyTorch hooks that enable the necessary casts are at the low-level
functional interface to PyTorch, so even custom layers will work with
amp, so long as they are built out of PyTorch functions and methods.

In particular, amp hooks into all of the following:
- Functions in the top-level `torch` namespace
- Functions in the `torch.nn.functional` namespace
- Methods on `Tensor` objects (GPU only, fp16 and fp32)
- Custom support for RNNs, even though they have no direct functional
  interface:
 - Recurrent cells: `torch.nn.{RNNCell,  LSTMCell, GRUCell}`
 - Recurrent layers: `torch.nn.{RNN, LSTM, GRU}`

In a few limited cases, amp needs help finding custom user-defined
functions that use low-level PyTorch features. In those cases, a
simple annotation is sufficient; this is described below.

## Installation and Requirements
amp is developed on Python 3.6 and PyTorch 0.4. It takes care to be
backwards-compatible with PyTorch 0.3, but users are _highly_
encouraged to upgrade.

amp is installed during normal apex installation, so refer to the
top-level README for more on installation.

## Usage and Getting Started

In the normal case, using amp requires adding two lines of code (and
an import). The first enables amp, so that it can hook into all the
relevant PyTorch functions. The second tells it where backpropagation
occurs so that it can properly scale the loss and clear internal
per-iteration state.

#### 1. Enable amp
```python
from apex import amp
amp_handle = amp.enable()
```

`amp.enable()` takes two arguments, and the defaults are _highly_
recommended. The first, `enable_caching` (default=True), indicates
whether amp should cache fp16 casts of model parameters on a
per-iteration basis. This prevents things like RNN cells used inside a
loop from casting their weight matrices over and over. The second,
`verbose` (default=False) toggles whether to print out every cast that
occurs. Useful for debugging, mostly.

#### 2. Wrap backpropagation

Nearly all PyTorch training scripts have a loops that looks like:

```python
# ... do a bunch of stuff to compute a loss
loss.backward()
optimizer.step()
# ...finish the iteration
```

To use amp, you need only tell it where backprop occurs:

```python
# ... same as before
with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
optimizer.step()
# ... same as before
```

This context manager allows amp to:
1. Use automatic loss scaling to best use fp16 range
2. Clear its cache of casted parameters before the next optimizer step

Note that it is _possible_ to use amp without step 2. In which case,
you will not get automatic loss scaling, nor is it safe to
`enable_caching`. (Power user note: you can manually clear the cache
after each optimizer step with `amp_handle._clear_cache()`.)

## Annotating User Functions

Nearly all PyTorch user code needs nothing more than steps one and two
above to use amp. After all, custom layers are built out of simpler
PyTorch components, and amp already can see those.

However, any custom C++ or CUDA code is outside of amp's (default)
view of things. For example, suppose I implemented a new recurrent
cell called a "forgetful recurrent unit" that calls directly into a
CUDA backend:

```python
def fru(input, hidden, weight, bias):
    # ... call to CUDA code
```

amp exposes two functions to handle this case: `register_fp16` and
`register_fp32`. These add the given function to the white or
blacklist, respectively. You can use them as a decorator:
```python
@amp.register_fp16
def fru(input, hidden, weight, bias):
    # ...
```
or as a library call:
```python
from apex import amp
amp.register_fp16(custom_module.fru)
amp.enable()
```

Note that the function must be registered before the call to
`amp.enable()`. The library call makes this simple. If the function is
annotated, then you must ensure its module is loaded before the call
to `amp.enable()`. Furthermore, this does not (yet) work with class
methods, only free functions.
