// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

#include <gds.h>

// torch
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

// cuda
#include <cuda_runtime.h>
#include <cufile.h>

// file io
#include <fcntl.h>

namespace apex::contrib::gds {

// POSIX
template <
    class T,
    typename std::enable_if<std::is_integral<T>::value, std::nullptr_t>::type =
        nullptr>
std::string cuFileGetErrorString(T status) {
  status = std::abs(status);
  return IS_CUFILE_ERR(status) ? std::string(CUFILE_ERRSTR(status))
                               : std::string(std::strerror(errno));
}

// CUfileError_t
template <
    class T,
    typename std::enable_if<!std::is_integral<T>::value, std::nullptr_t>::type =
        nullptr>
std::string cuFileGetErrorString(T status) {
  std::string errStr = cuFileGetErrorString(static_cast<int>(status.err));
  if (IS_CUDA_ERR(status))
    errStr.append(".").append(cudaGetErrorString(static_cast<cudaError_t>(status.cu_err)));
  return errStr;
}

void load_data(torch::Tensor& tensor, std::string& filename) {
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  int fd = -1;
  ssize_t ret = -1;
  void* dataPtr = tensor.data_ptr();
  const size_t size = tensor.nbytes();
  CUfileError_t status;

  CUfileDescr_t cf_descr;
  CUfileHandle_t cf_handle;

  // Open the binary file
  fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
  TORCH_CHECK(fd >= 0, "fcntl cannot open file: ", filename);

  // Register cuFile handle
  memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  status = cuFileHandleRegister(&cf_handle, &cf_descr);
  if (status.err != CU_FILE_SUCCESS) {
    close(fd);
    fd = -1;
    TORCH_CHECK(false, "cuFileHandleRegister failed: ", cuFileGetErrorString(status));
  }

  // Read the binary file
  ret = cuFileRead(cf_handle, (void*)dataPtr, size, 0, 0);
  if (ret < 0) {
    cuFileHandleDeregister(cf_handle);
    close(fd);
    TORCH_CHECK(false, "cuFileWrite failed: ", cuFileGetErrorString(ret));
  }

  // Deregister cuFile handle and close the file
  cuFileHandleDeregister(cf_handle);
  close(fd);
}

void save_data(torch::Tensor& tensor, std::string& filename) {
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  int fd = -1;
  ssize_t ret = -1;
  void* dataPtr = tensor.data_ptr();
  const size_t size = tensor.nbytes();
  CUfileError_t status;

  CUfileDescr_t cf_descr;
  CUfileHandle_t cf_handle;

  // Opens a file to write
  fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_DIRECT, 0664);
  TORCH_CHECK(fd >= 0, "fcntl cannot open file: ", filename);

  // Register cuFile handle
  memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  status = cuFileHandleRegister(&cf_handle, &cf_descr);
  if (status.err != CU_FILE_SUCCESS) {
    close(fd);
    fd = -1;
    TORCH_CHECK(false, "cuFileHandleRegister failed: ", cuFileGetErrorString(status));
  }

  // Register device memory
  status = cuFileBufRegister(dataPtr, size, 0);
  if (status.err != CU_FILE_SUCCESS) {
    TORCH_CHECK(false, "cuFileBufRegister failed: ", cuFileGetErrorString(status));
  }

  // Write device memory contents to the file
  ret = cuFileWrite(cf_handle, dataPtr, size, 0, 0);
  if (ret < 0) {
    cuFileBufDeregister(dataPtr);
    cuFileHandleDeregister(cf_handle);
    close(fd);
    TORCH_CHECK(false, "cuFileWrite failed: ", cuFileGetErrorString(ret));
  }

  // Deregister the device memory
  status = cuFileBufDeregister(dataPtr);

  // Deregister cuFile handle and close the file
  cuFileHandleDeregister(cf_handle);
  close(fd);

  TORCH_CHECK(status.err == CU_FILE_SUCCESS, "cuFileBufDeregister failed:", cuFileGetErrorString(status));
}


// Just for benchmarking purposes

void load_data_no_gds(torch::Tensor& tensor, std::string& filename) {
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  void* dataPtrCpu = nullptr;
  void* dataPtr = tensor.data_ptr();
  const size_t numel = tensor.numel();
  const size_t element_size = tensor.element_size();
  const size_t size = tensor.nbytes();
  dataPtrCpu = malloc(size);
  TORCH_CHECK(dataPtrCpu != nullptr, "malloc failed");

  FILE *fp_output;
  fp_output = fopen(filename.c_str(), "rb");
  TORCH_CHECK(fp_output != nullptr, "stdio cannot fopen file: ", filename);
  const size_t count = fread(dataPtrCpu, element_size, numel, fp_output);
  TORCH_CHECK(count == numel, "fread failed");
  fclose(fp_output);
  C10_CUDA_CHECK(cudaMemcpy(dataPtr, dataPtrCpu, size, cudaMemcpyHostToDevice));
  free(dataPtrCpu);
}

void save_data_no_gds(torch::Tensor& tensor, std::string& filename) {
  c10::cuda::CUDAGuard gpuGuard(tensor.device());

  void* dataPtrCpu = nullptr;
  void* dataPtr = tensor.data_ptr();
  const size_t numel = tensor.numel();
  const size_t element_size = tensor.element_size();
  const size_t size = tensor.nbytes();
  dataPtrCpu = malloc(size);
  TORCH_CHECK(dataPtrCpu != nullptr, "malloc failed");
  C10_CUDA_CHECK(cudaMemcpy(dataPtrCpu, dataPtr, size, cudaMemcpyDeviceToHost));

  FILE *fp_output;
  fp_output = fopen(filename.c_str(), "wb");
  TORCH_CHECK(fp_output != nullptr, "stdio cannot fopen file: ", filename);
  const size_t count = fwrite(dataPtrCpu, element_size, numel, fp_output);
  TORCH_CHECK(count == numel, "fwrite failed");
  fclose(fp_output);
  free(dataPtrCpu);
}

} // namespace apex::contrib::gds
