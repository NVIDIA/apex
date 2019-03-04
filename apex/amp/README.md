# amp: Automatic Mixed Precision

## This README documents the deprecated (pre-unified) API.

## Documentation for the current unified API can be found [here](https://nvidia.github.io/apex/)

amp is an experimental tool to enable mixed precision training in
PyTorch with extreme simplicity and overall numerical safety. It
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

In the common case, using amp requires adding two lines of code (and
an import). The first enables amp, so that it can hook into all the
relevant PyTorch functions. The second tells it where backpropagation
occurs so that it can properly scale the loss and clear internal
per-iteration state.

#### 1. Enable amp
```python
from apex import amp
amp_handle = amp.init()
```

`amp.init()` takes three (optional) arguments. The most useful is
`enabled` (default=True), which simplifies command-line arguments. If
False, then everything amp does will be a zero-overhead pass-through
-- i.e., your code will run as-is.

For the other two options, the defaults are _highly_ recommended. The
first, `enable_caching` (default=True), indicates whether amp should
cache fp16 casts of model parameters on a per-iteration basis. This
prevents things like RNN cells used inside a loop from casting their
weight matrices over and over. The second, `verbose` (default=False)
toggles whether to print out every cast that occurs. Useful for
debugging, mostly.

#### 2. Wrap backpropagation

Nearly all PyTorch training scripts have a loop that looks like:

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

## Multiple Optimizers or Backward Passes

Step (2) from the previous section works when you have one PyTorch
optimizer and a single `loss.backward()` for each iteration. Some
models are more complex with:
- Multiple optimizer objects (over different parameters)
- Multiple backward passes for each iteration, taking advantage of
  PyTorch's gradient accumulation

To work with such models, amp requires you to explicitly wrap each
optimizer and indicate if it will have more than one backward pass
per-iteration.

#### Explicitly wrapping optimizers

If you have more than one optimizer, then you must explicitly wrap
each. (You can also do so with a single optimizer.) First, wrap the
optimizer after initializing amp:

```python
optimizer = # ... some optimizer

amp_handle = amp.init()
optimizer = amp_handle.wrap_optimizer(optimizer)
```

Second, use `optimizer.scale_loss(...)` to indicate where backprop
occurs:

```python
with optimizer.scale_loss(loss) as scaled_loss:
    scaled_loss.backward()
optimizer.step()
# ...
```

In essence, `amp_handle.scale_loss(loss, optimizer)` is syntactic
sugar for first wrapping the optimizer and then calling
`optimizer.scale_loss(loss)` in the single-optimizer case. But in the
multi-optimizer case, you must wrap each optimizer individually.

#### Handling multiple backward passes

PyTorch accumulates parameter gradients between calls to
`zero_grad()`, so it is possible to perform multiple backward passes
before making a parameter update:

```python
optimizer.zero_grad()
loss1 = ComputeLoss1(model)
loss1.backward()
# ...
loss2 = ComputeLoss2(model)
loss2.backward()
# ...
optimizer.step() # has gradient contributions from both backward passes
```

The amp optimizer wrapper supports an additional argument `num_loss`
to work with code like this:

```python
amp_handle = amp.init()
optimizer = amp_handle.wrap_optimizer(optimizer, num_loss=2)
# ...
optimizer.zero_grad()
loss1 = ComputeLoss1(model)
with optimizer.scale_loss(loss1) as scaled_loss:
    scaled_loss.backward()
# ...
loss2 = ComputeLoss2(model)
with optimizer.scale_loss(loss2) as scaled_loss:
    scaled_loss.backward()
# ...
optimizer.step()
```

## Annotating User Functions

Nearly all PyTorch user code needs nothing more than the two steps
above to use amp. After all, custom layers are built out of simpler
PyTorch components, and amp already can see those.

However, any custom C++ or CUDA code is outside of amp's (default)
view of things. For example, suppose I implemented a new recurrent
cell called a "forgetful recurrent unit" that calls directly into a
CUDA backend:

```python
from backend import FRUBackend

def fru(input, hidden, weight, bias):
    # call to CUDA code
    FRUBackend(input, hidden, weight, bias)
```

In this case, it is possible to get a runtime type mismatch. For
example, you might have `input` in fp16, and `weight` in fp32, and amp
doesn't have the visibility to insert an appropriate cast.

amp exposes two ways to handle "invisible" backend code: function
annotations and explicit registration.

#### Function annotation

The first way to handle backend code is a set of function annotations:

- `@amp.half_function`
- `@amp.float_function`
- `@amp.promote_function`

These correspond to:

- Cast all arguments to fp16
- Cast all argumnets fo fp32
- If there are any type mismatches, cast everything to the widest type

In our example, we believe that the FRU unit is fp16-safe and will get
performance gains from casting its arguments to fp16, so we write:

```python
@amp.half_function
def fru(input, hidden, weight, bias):
    #...
```

#### Explicit registration

The other way to handle backend code is with explicit function
registration:

- `amp.register_half_function(module, function_name)`
- `amp.register_float_function(module, function_name)`
- `amp.register_promote_function(module, function_name)`

When using this API, `module` is the containing class or module for
the function, and `function_name` is the _string_ name of the
function. Note that the function must be registered before the call to
`amp.init()`.

For our FRU unit, we can register the backend function directly:

```python
import backend

amp.register_half_function(backend, 'FRUBackend')
amp.init()
```
