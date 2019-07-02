# amp: Automatic Mixed Precision

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
`amp.initalize()`.

For our FRU unit, we can register the backend function directly:

```python
import backend

amp.register_half_function(backend, 'FRUBackend')
```
