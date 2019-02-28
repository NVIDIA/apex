.. role:: hidden
    :class: hidden-section

apex.amp
===================================

This page documents Amp (Automatic Mixed Precision) 1.0, a tool to enable Tensor Core-accelerated
training in only 3 lines of Python.

Amp allows users to easily experiment with different pure and mixed precision modes, including
pure FP16 training and pure FP32 training.  Commonly-used default modes are chosen by
selecting an "optimization level" or ``opt_level``; each ``opt_level`` establishes a set of
properties that govern Amp's implementation of pure or mixed precision training.
Finer-grained control of how a given ``opt_level`` behaves can be achieved by also passing values for
particular properties directly to ``amp.initialize``.  These manually specified values will
override the defaults established by the ``opt_level``.  If you attempt to override a property
that does not make sense for the current ``opt_level``, Amp will raise an error with an explanation.

Users **should not** manually cast their model or data to ``.half()``, regardless of what ``opt_level``
or properties are chosen.  Amp intends that users start with an existing default (FP32) script,
add the three lines corresponding to the Amp 1.0 API, and begin training with mixed precision.
Amp can also be disabled, in which case the original script will behave exactly as it used to.
In this way, there's no risk adhering to the Amp 1.0 API, and a lot of potential performance benefit.

Example::
        model = torch.nn.Linear(D_in, D_out).cuda().half()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        ...
        # loss.backward() becomes:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        ...

.. automodule:: apex.amp
.. currentmodule:: apex.amp

.. autofunction:: initialize

.. autofunction:: scale_loss



Legacy documentation for the old "Amp" API (equivalent to ``opt_level="O1"`` in the new Amp 1.0 API) can be found on the Github README:  https://github.com/NVIDIA/apex/tree/master/apex/amp.
