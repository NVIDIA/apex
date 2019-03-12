.. role:: hidden
    :class: hidden-section

Advanced Amp Usage
===================================

GANs
----

GANs are an interesting synthesis of several topics below.  A `comprehensive example`_
is under construction.

.. _`comprehensive example`:
    https://github.com/NVIDIA/apex/tree/master/examples/dcgan

Gradient clipping
-----------------
Amp calls the params owned directly by the optimizer's ``param_groups`` the "master params."

These master params may be fully or partially distinct from ``model.parameters()``.
For example, with `opt_level="O2"`_, ``amp.initialize`` casts most model params to FP16,
creates an FP32 master param outside the model for each newly-FP16 model param,
and updates the optimizer's ``param_groups`` to point to these FP32 params.

The master params owned by the optimizer's ``param_groups`` may also fully coincide with the
model params, which is typically true for ``opt_level``\s ``O0``, ``O1``, and ``O3``.

In all cases, correct practice is to clip the gradients of the params that are guaranteed to be
owned **by the optimizer's** ``param_groups``, instead of those retrieved via ``model.parameters()``.

Also, if Amp uses loss scaling, gradients must be clipped after they have been unscaled
(which occurs during exit from the ``amp.scale_loss`` context manager).

The following pattern should be correct for any ``opt_level``::

    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
        # Gradients are unscaled during context manager exit.
    # Now it's safe to clip.  Replace
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    # with
    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
    # or
    torch.nn.utils.clip_grad_value_(amp.master_params(optimizer), max_)

Note the use of the utility function ``amp.master_params(optimizer)``,
which returns a generator-expression that iterates over the
params in the optimizer's ``param_groups``.

Also note that ``clip_grad_norm_(amp.master_params(optimizer), max_norm)`` is invoked
*instead of*, not *in addition to*, ``clip_grad_norm_(model.parameters(), max_norm)``.

.. _`opt_level="O2"`:
    https://nvidia.github.io/apex/amp.html#o2-fast-mixed-precision

Custom/user-defined autograd functions
--------------------------------------

The old Amp API for `registering user functions`_ is still considered correct.  Functions must
be registered before calling ``amp.initialize``.

.. _`registering user functions`:
    https://github.com/NVIDIA/apex/tree/master/apex/amp#annotating-user-functions

Forcing particular layers/functions to a desired type
-----------------------------------------------------

I'm still working on a generalizable exposure for this that won't require user-side code divergence
across different ``opt-level``\ s.

Multiple models/optimizers
--------------------------

``amp.initialize``'s optimizer argument may be a single optimizer or a list of optimizers,
as long as the output you accept has the same type.
Similarly, the ``model`` argument may be a single model or a list of models, as long as the accepted
output matches.  The following calls are all legal::

    model, optim = amp.initialize(model, optim,...)
    model, [optim1, optim2] = amp.initialize(model, [optim1, optim2],...)
    [model1, model2], optim = amp.initialize([model1, model2], optim,...)
    [model1, model2], [optim1, optim2] = amp.initialize([model1, model2], [optim1, optim2],...)

Whenever you invoke a backward pass, the optimizer you should pass to ``amp.scaled_loss`` is whatever
optimizer owns the parameters for which this particular backward pass is creating gradients.

Multiple backward passes per iteration
--------------------------------------

If you want to accumulate gradients from multiple losses for the params owned by a given optimizer,
you must invoke ``with amp.scale_loss(..., delay_unscale=True)`` for all backward passes except
the last::

    # delay_unscale=True for the first two losses
    with amp.scale_loss(loss1, optimizer, delay_unscale=True) as scaled_loss:
        scaled_loss.backward()
    with amp.scale_loss(loss2, optimizer, delay_unscale=True) as scaled_loss:
        scaled_loss.backward()
    # Don't delay_unscale for the final loss 
    with amp.scale_loss(loss3, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()


Gradient accumulation across iterations
---------------------------------------

Pass ``delay_unscale=True`` to ``amp.scale_loss`` until you're ready to ``step()``::

    if iter%iters_to_accumulate == 0:
        # Every iters_to_accumulate iterations, unscale and step
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    else:
        # Otherwise, just accumulate gradients, don't unscale or step. 
        with amp.scale_loss(loss, optimizer, delay_unscale=True) as scaled_loss:
            scaled_loss.backward()
