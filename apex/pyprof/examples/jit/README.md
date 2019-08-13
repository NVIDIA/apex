*As of this writing, these examples do not work
because of changes being proposed in PyTorch.*

There are two ways to use PyTorch JIT
 - Scripting
 - Tracing

In addition, we can JIT a
 - Stand alone function
 - Class / class method

This directory has an example for each of the 4 cases.
Intercepting (monkey patching) JITted code has a few extra steps,
which are explained through comments.
