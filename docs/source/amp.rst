.. role:: hidden
    :class: hidden-section

apex.amp
===================================

Unified API
-----------

This page documents the updated API for Amp (Automatic Mixed Precision),
a tool to enable Tensor Core-accelerated training in only 3 lines of Python.

Amp allows users to easily experiment with different pure and mixed precision modes, including
pure FP16 training and pure FP32 training.  Commonly-used default modes are chosen by
selecting an "optimization level" or ``opt_level``; each ``opt_level`` establishes a set of
properties that govern Amp's implementation of pure or mixed precision training.
Finer-grained control of how a given ``opt_level`` behaves can be achieved by passing values for
particular properties directly to ``amp.initialize``.  These manually specified values will
override the defaults established by the ``opt_level``.  If you attempt to override a property
that does not make sense for the current ``opt_level``, Amp will raise an error with an explanation.

Users **should not** manually cast their model or data to ``.half()``, regardless of what ``opt_level``
or properties are chosen.  Amp intends that users start with an existing default (FP32) script,
add the three lines corresponding to the Amp API, and begin training with mixed precision.
Amp can also be disabled, in which case the original script will behave exactly as it used to.
In this way, there's no risk adhering to the Amp API, and a lot of potential performance benefit.

Example::

        # Declare model and optimizer as usual
        model = torch.nn.Linear(D_in, D_out).cuda().half()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # Allow Amp to perform casts as required by the opt_level
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        ...
        # loss.backward() becomes:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        ...

A `runnable, comprehensive Imagenet example`_ demonstrating good practices can be found
on the Github page.

GANs are a tricky case that many people have requested.  A `comprehensive DCGAN example`_
is under construction.

``opt_level``\ s and Properties
-------------------------------

.. _`runnable, comprehensive Imagenet example`:
    https://github.com/NVIDIA/apex/tree/master/examples/imagenet

.. _`comprehensive DCGAN example`:
    https://github.com/NVIDIA/apex/tree/master/examples/dcgan

.. automodule:: apex.amp
.. currentmodule:: apex.amp

.. autofunction:: initialize

.. autofunction:: scale_loss

.. autofunction:: master_params

Advanced use cases
------------------

The new Amp API supports gradient accumulation across iterations,
multiple backward passes per iteration, multiple models/optimizers,
and custom/user-defined autograd functions.  Gradient clipping and GANs also
require special treatment, but this treatment does not need to change
for different ``opt_level``\ s.  Further details can be found here:

.. toctree::
   :maxdepth: 1

   advanced

Transition guide for old API users
----------------------------------

We strongly encourage moving to the new Amp API, because it's more versatile, easier to use, and future proof.  The original :class:`FP16_Optimizer` and the old "Amp" API are deprecated, and subject to removal at at any time.

**For users of the old "Amp" API**

In the new API, ``opt-level O1`` performs the same patching of the Torch namespace as the old Amp API.
However, the new API allows choosing static or dynamic loss scaling, while the old API only allowed dynamic loss scaling.

In the new API, the old call to ``amp_handle = amp.init()``, and the returned ``amp_handle``, are no
longer exposed or necessary.  The new ``amp.initialize()`` does the duty of ``amp.init()`` (and more).
Therefore, any existing calls to ``amp_handle = amp.init()`` should be deleted.

The functions formerly exposed through ``amp_handle`` are now free
functions accessible through the ``amp`` module.

The backward context manager must be changed accordingly::

    # old API
    with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    ->
    # new API
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

For now, the deprecated "Amp" API documentation can still be found on the Github README:  https://github.com/NVIDIA/apex/tree/master/apex/amp.  The old API calls that `annotate user functions`_ to run
with a particular precision are still honored by the new API.

.. _`annotate user functions`:
    https://github.com/NVIDIA/apex/tree/master/apex/amp#annotating-user-functions


**For users of the old FP16_Optimizer**

``opt-level O2`` is equivalent to :class:`FP16_Optimizer` with ``dynamic_loss_scale=True``.
Once again, the backward pass must be changed to the unified version::

    optimizer.backward(loss)
    ->
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

One annoying aspect of FP16_Optimizer was that the user had to manually convert their model to half
(either by calling ``.half()`` on it, or using a function or module wrapper from
``apex.fp16_utils``), and also manually call ``.half()`` on input data.  **Neither of these are
necessary in the new API.  No matter what --opt-level
you choose, you can and should simply build your model in the default FP32 format.**  The new Amp
API will perform the right conversions during
``model, optimizer = amp.initialize(model, optimizer, opt_level=....)`` based on the ``--opt-level``
and any overridden flags.  Floating point input data may be FP32 or FP16, but you may as well just
let it be FP16, because the ``model`` returned by ``amp.initialize`` will have its ``forward``
method patched to cast the input data appropriately.

.. note::
    Aside from the call to ``amp.initialize`` itself, it's never necessary to manually cast
    your model or data with the new API.  Therefore, a script that adheres to the new API
    can switch between different ``opt-level``\ s without having to make any other changes.
