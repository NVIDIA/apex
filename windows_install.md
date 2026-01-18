# Build Fixes for NVIDIA Apex on Windows 11 (CUDA 12.8 / MSVC 2022)

## Installation Command

Make sure you run below commands in **x64 Native Tools Command Prompt for VS 2022** (use search in the win11 to find it). Before install it, make sure your environment has the necessary dependencies like `Pytorch` and `ninja`.

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
set APEX_CPP_EXT=1
set APEX_CUDA_EXT=1
set DISTUTILS_USE_SDK=1
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation ./
```

---

> **Note:** Building NVIDIA Apex on Windows is challenging and may find different errors on different devices. This guide documents a successful build on Win11 RTX5070 (sm_120) with CUDA 12.8.

---

## Build Environment

| Component | Version |
|-----------|---------|
| **OS** | Windows 11 |
| **CUDA Toolkit** | 12.8 (Blackwell / SM_100 / SM_120) |
| **CUDA Path** | `E:\CUDA128` |
| **Compiler** | MSVC 2022 (Visual Studio Build Tools) |
| **Python** | 3.10 |
| **PyTorch** | 2.9.1+cu128 |
| **Build Flags** | `APEX_CPP_EXT=1`, `APEX_CUDA_EXT=1` |

### NVCC Version Info

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:38:46_Pacific_Standard_Time_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0
```

---

## Summary of Changes

This patch addresses **three primary categories** of build failures encountered on Windows:

1. Standard type definitions
2. MSVC-specific compiler flags for memory alignment
3. Explicit library linking for cuBLAS

---

## 1. `setup.py` Configuration

### Changes

Added `libraries=["cublas", "cublasLt"]` and `extra_compile_args` with `-D_DISABLE_EXTENDED_ALIGNED_STORAGE` to several CUDA extensions.

### Affected Extensions

- `mlp_cuda`
- `fused_dense_cuda`
- `fused_weight_gradient_mlp_cuda`
- *(And potentially others using cuBLAS or aligned storage)*

### Code Diff

```python
ext_modules.append(
    CUDAExtension(
        name="module_name",
        sources=["..."],
        # Fix 1: Explicitly link cuBLAS for Windows
        libraries=["cublas", "cublasLt"], 
        extra_compile_args={
            # Fix 2: Disable extended aligned storage to fix VS2019+ static assertion errors
            "cxx": ["-O3", "-D_DISABLE_EXTENDED_ALIGNED_STORAGE"],
            "nvcc": ["-O3", "-D_DISABLE_EXTENDED_ALIGNED_STORAGE", ...],
        },
    )
)
```

### Reasoning

| Issue | Explanation |
|-------|-------------|
| **Linker Errors (`LNK2001`)** | Unlike Linux, the Windows build environment does not automatically link `cublas.lib` and `cublasLt.lib` when these headers are used. Explicit linking resolves unresolved external symbols for `cublasGemmEx`, `cublasLtMatmul`, etc. |
| **Alignment Errors** | Visual Studio 2017 (15.8 update) and later changed how `std::aligned_storage` works, causing compliance standard errors with older CUDA headers. The flag `_DISABLE_EXTENDED_ALIGNED_STORAGE` restores the necessary behavior for compilation to succeed. |

---

## 2. Source Code Fixes (`csrc/`)

### A. Type Definition Fix (`uint`)

**File:** `csrc/mlp_cuda.cu`

**Change:** Replaced `uint` with `unsigned int`.

**Reasoning:** The type alias `uint` is standard in Linux system headers but is **not defined** by default in the MSVC (Windows) environment. Using the standard C++ type `unsigned int` ensures cross-platform compatibility.

---

### B. Device Function Compatibility (`isfinite`)

**Files:**
- `csrc/multi_tensor_scale_kernel.cu`
- `csrc/multi_tensor_axpby_kernel.cu`

**Change:** Replaced the `isfinite()` check with a robust floating-point check using `fabsf`. Affected variables including `r_in[ii]`, `r_x[ii]` and `r_y[ii]`.

```cpp
// Before
finite = finite && (isfinite(r_in[ii])); ...

// After
finite = finite && (fabsf((float)r_in[ii]) <= 3.40282e+38f); ... 
// Checks if value is within finite float range
```

**Reasoning:** On Windows NVCC, `isfinite` often resolves to the host-only C++ standard library function (`std::isfinite`) rather than the device intrinsic, causing a *"calling a host function from a device function"* error. Replacing it with `fabsf` (which is correctly mapped to a device intrinsic) bypasses this restriction while maintaining logical correctness.

---




## License

Follow the original [NVIDIA Apex License](https://github.com/NVIDIA/apex/blob/master/LICENSE).