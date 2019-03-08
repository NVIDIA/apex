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
If Amp uses master params distinct from the model params,
then the params ``step()``\ ed by the optimizer are the master params,
and it is the master gradients (rather than the model gradients) that must be clipped.

If Amp is not using master params distinct from the model params, then the optimizer
directly steps the model params, and the model grads must be clipped.

In both cases, correct practice is to clip the gradients of the params that are about to be stepped **by the optimizer** (which may be distinct from ``model.parameters()``).

Also, if Amp uses loss scaling, gradients must be clipped after they have been unscaled.

The following pattern accounts for all possibilities, and should be correct for
any ``opt_level``::

    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
        # Gradients are unscaled during context manager exit.
    # Now it's safe to clip:
    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
    # or
    torch.nn.utils.clip_grad_value_(amp.master_params(optimizer), max_)

Note the use of the utility function ``amp.master_params(optimizer)``,
which returns a generator-expression that iterates over the
params that the optimizer steps (master params if enabled, otherwise model params).

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
