# Fast Multihead Attention 

This implementation has too main features :
* A C++ implementation to avoid the CPU overheads of Pytorch found with smaller batch sizes.
* The removal of all copies and transposes found in standard implementations of Multihead Attention.

|                                     | Python Version | C++ Version |
| :---------------------------------- | :------------: | :---------: |
| Layer Norm and Residual Add Variant | X              | X           |
| Includes Linear Biases              | X              |             |
| Reduces CPU Overheads               |                | X           |
| Fuses masking with Softmax          |                | X           |
| Removes Transposes and Copies       | X              | X           |
