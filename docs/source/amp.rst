.. role:: hidden
    :class: hidden-section

apex.amp
===================================

This page documents the updated API for Amp (Automatic Mixed Precision),
a tool to enable Tensor Core-accelerated training in only 3 lines of Python.

A `runnable, comprehensive Imagenet example`_ demonstrating good practices can be found
on the Github page.

GANs are a tricky case that many people have requested.  A `comprehensive DCGAN example`_
is under construction.

If you already implemented Amp based on the instructions below, but it isn't behaving as expected,
please review `Advanced Amp Usage`_ to see if any topics match your use case.  If that doesn't help,
`file an issue`_.

.. _`file an issue`:
    https://github.com/NVIDIA/apex/issues

``opt_level``\ s and Properties
-------------------------------

Amp allows users to easily experiment with different pure and mixed precision modes.
Commonly-used default modes are chosen by
selecting an "optimization level" or ``opt_level``; each ``opt_level`` establishes a set of
properties that govern Amp's implementation of pure or mixed precision training.
Finer-grained control of how a given ``opt_level`` behaves can be achieved by passing values for
particular properties directly to ``amp.initialize``.  These manually specified values
override the defaults established by the ``opt_level``.

Example::

        # Declare model and optimizer as usual, with default (FP32) precision
        model = torch.nn.Linear(D_in, D_out).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # Allow Amp to perform casts as required by the opt_level
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        ...
        # loss.backward() becomes:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        ...

Users **should not** manually cast their model or data to ``.half()``, regardless of what ``opt_level``
or properties are chosen.  Amp intends that users start with an existing default (FP32) script,
add the three lines corresponding to the Amp API, and begin training with mixed precision.
Amp can also be disabled, in which case the original script will behave exactly as it used to.
In this way, there's no risk adhering to the Amp API, and a lot of potential performance benefit.

.. note::
    Because it's never necessary to manually cast your model (aside from the call ``amp.initialize``)
    or input data, a script that adheres to the new API
    can switch between different ``opt-level``\ s without having to make any other changes.

.. _`runnable, comprehensive Imagenet example`:
    https://github.com/NVIDIA/apex/tree/master/examples/imagenet

.. _`comprehensive DCGAN example`:
    https://github.com/NVIDIA/apex/tree/master/examples/dcgan

.. _`Advanced Amp Usage`:
    https://nvidia.github.io/apex/advanced.html

Properties
**********

Currently, the under-the-hood properties that govern pure or mixed precision training are the following:

- ``cast_model_type``:  Casts your model's parameters and buffers to the desired type.
- ``patch_torch_functions``: Patch all Torch functions and Tensor methods to perform Tensor Core-friendly ops like GEMMs and convolutions in FP16, and any ops that benefit from FP32 precision in FP32.
- ``keep_batchnorm_fp32``:  To enhance precision and enable cudnn batchnorm (which improves performance), it's often beneficial to keep batchnorm weights in FP32 even if the rest of the model is FP16.
- ``master_weights``:  Maintain FP32 master weights to accompany any FP16 model weights.  FP32 master weights are stepped by the optimizer to enhance precision and capture small gradients.
- ``loss_scale``:  If ``loss_scale`` is a float value, use this value as the static (fixed) loss scale.  If ``loss_scale`` is the string ``"dynamic"``, adaptively adjust the loss scale over time.  Dynamic loss scale adjustments are performed by Amp automatically.

Again, you often don't need to specify these properties by hand.  Instead, select an ``opt_level``,
which will set them up for you.  After selecting an ``opt_level``, you can optionally pass property
kwargs as manual overrides.

If you attempt to override a property that does not make sense for the selected ``opt_level``,
Amp will raise an error with an explanation.  For example, selecting ``opt_level="O1"`` combined with
the override ``master_weights=True`` does not make sense.  ``O1`` inserts casts
around Torch functions rather than model weights.  Data, activations, and weights are recast
out-of-place on the fly as they flow through patched functions.  Therefore, the model weights themselves
can (and should) remain FP32, and there is no need to maintain separate FP32 master weights.

``opt_level``\ s
****************

Recognized ``opt_level``\ s are ``"O0"``, ``"O1"``, ``"O2"``, and ``"O3"``.

``O0`` and ``O3`` are not true mixed precision, but they are useful for establishing accuracy and
speed baselines, respectively.

``O1`` and ``O2`` are different implementations of mixed precision.  Try both, and see
what gives the best speedup and accuracy for your model.

``O0``:  FP32 training
^^^^^^^^^^^^^^^^^^^^^^
Your incoming model should be FP32 already, so this is likely a no-op.
``O0`` can be useful to establish an accuracy baseline.

| Default properties set by ``O0``:
| ``cast_model_type=torch.float32``
| ``patch_torch_functions=False``
| ``keep_batchnorm_fp32=None`` (effectively, "not applicable," everything is FP32)
| ``master_weights=False``
| ``loss_scale=1.0``
|
|

``O1``:  Mixed Precision (recommended for typical use)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Patch all Torch functions and Tensor methods to cast their inputs according to a whitelist-blacklist
model.  Whitelist ops (for example, Tensor Core-friendly ops like GEMMs and convolutions) are performed
in FP16.  Blacklist ops that benefit from FP32 precision (for example, softmax)
are performed in FP32.  ``O1`` also uses dynamic loss scaling, unless overridden.

| Default properties set by ``O1``:
| ``cast_model_type=None`` (not applicable)
| ``patch_torch_functions=True``
| ``keep_batchnorm_fp32=None`` (again, not applicable, all model weights remain FP32)
| ``master_weights=None`` (not applicable, model weights remain FP32)
| ``loss_scale="dynamic"``
|
|

``O2``:  "Almost FP16" Mixed Precision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``O2`` casts the model weights to FP16,
patches the model's ``forward`` method to cast input
data to FP16, keeps batchnorms in FP32, maintains FP32 master weights,
updates the optimizer's ``param_groups`` so that the ``optimizer.step()``
acts directly on the FP32 weights (followed by FP32 master weight->FP16 model weight
copies if necessary),
and implements dynamic loss scaling (unless overridden).
Unlike ``O1``, ``O2`` does not patch Torch functions or Tensor methods.

| Default properties set by ``O2``:
| ``cast_model_type=torch.float16``
| ``patch_torch_functions=False``
| ``keep_batchnorm_fp32=True``
| ``master_weights=True``
| ``loss_scale="dynamic"``
|
|

``O3``:  FP16 training
^^^^^^^^^^^^^^^^^^^^^^
``O3`` may not achieve the stability of the true mixed precision options ``O1`` and ``O2``.
However, it can be useful to establish a speed baseline for your model, against which
the performance of ``O1`` and ``O2`` can be compared.  If your model uses batch normalization,
to establish "speed of light" you can try ``O3`` with the additional property override
``keep_batchnorm_fp32=True`` (which enables cudnn batchnorm, as stated earlier).

| Default properties set by ``O3``:
| ``cast_model_type=torch.float16``
| ``patch_torch_functions=False``
| ``keep_batchnorm_fp32=False``
| ``master_weights=False``
| ``loss_scale=1.0``
|
|

Unified API
-----------

.. automodule:: apex.amp
.. currentmodule:: apex.amp

.. autofunction:: initialize

.. autofunction:: scale_loss

.. autofunction:: master_params

Advanced use cases
------------------

The unified Amp API supports gradient accumulation across iterations,
multiple backward passes per iteration, multiple models/optimizers,
custom/user-defined autograd functions, and custom data batch classes.  Gradient clipping and GANs also
require special treatment, but this treatment does not need to change
for different ``opt_level``\ s.  Further details can be found here:

.. toctree::
   :maxdepth: 1

   advanced

Transition guide for old API users
----------------------------------

We strongly encourage moving to the new Amp API, because it's more versatile, easier to use, and future proof.  The original :class:`FP16_Optimizer` and the old "Amp" API are deprecated, and subject to removal at at any time.

For users of the old "Amp" API
******************************

In the new API, ``opt-level O1`` performs the same patching of the Torch namespace as the old thing
called "Amp."
However, the new API allows static or dynamic loss scaling, while the old API only allowed dynamic loss scaling.

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


For users of the old FP16_Optimizer
***********************************

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
you choose, you can and should simply build your model and pass input data in the default FP32 format.**
The new Amp API will perform the right conversions during
``model, optimizer = amp.initialize(model, optimizer, opt_level=....)`` based on the ``--opt-level``
and any overridden flags.  Floating point input data may be FP32 or FP16, but you may as well just
let it be FP16, because the ``model`` returned by ``amp.initialize`` will have its ``forward``
method patched to cast the input data appropriately.
