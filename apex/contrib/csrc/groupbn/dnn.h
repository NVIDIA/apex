#ifndef DNN_H
#define DNN_H

#ifdef __HIP_PLATFORM_HCC__
#include <miopen/miopen.h>
#define DNN_STATUS_SUCCESS miopenStatusSuccess
#define DNN_DATA_HALF miopenHalf
#define DNN_TENSOR_FORMAT 0

using dnnTensorFormat_t = int;
using dnnDataType_t = miopenDataType_t;
using dnnStatus_t = miopenStatus_t;
using dnnTensorDescriptor_t = miopenTensorDescriptor_t;
#else
#include <cudnn.h>
#define DNN_STATUS_SUCCESS CUDNN_STATUS_SUCCESS
#define DNN_DATA_HALF CUDNN_DATA_HALF
#define DNN_TENSOR_FORMAT CUDNN_TENSOR_NHWC

using dnnTensorFormat_t = cudnnTensorFormat_t;
using dnnDataType_t = cudnnDataType_t;
using dnnStatus_t = cudnnStatus_t;
using dnnTensorDescriptor_t = cudnnTensorDescriptor_t;
#endif

#endif // DNN_H
