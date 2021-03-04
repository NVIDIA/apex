#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"

void ln_fwd_cuda(at::Tensor &y, at::Tensor &mu, at::Tensor &rsigma,
                 const at::Tensor &x, const at::Tensor &gamma,
                 const at::Tensor &beta, const float epsilon, const int rows, const int cols,
                 cudaStream_t stream);

void ln_bwd_cuda(at::Tensor &dx, at::Tensor &dgamma, at::Tensor &dbeta,
                 const at::Tensor &dw, const at::Tensor &x,
                 const at::Tensor &mu, const at::Tensor &rsigma,
                 const at::Tensor &gamma, const int rows, const int cols, cudaStream_t stream);


std::vector<at::Tensor> ln_fwd(const at::Tensor &x,      // BxSxhidden_size
                               const at::Tensor &gamma,   // hidden_size
                               const at::Tensor &beta,   // hidden_size
                               const float epsilon
) {

    TORCH_CHECK(x.is_cuda())
    TORCH_CHECK(gamma.is_cuda())
    TORCH_CHECK(beta.is_cuda())

    TORCH_CHECK(x.is_contiguous());
    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 2);

    const int rows = sizes[0];
    const int cols = sizes[1];

    auto dtype = x.scalar_type();

    TORCH_CHECK(gamma.dtype() == dtype);
    TORCH_CHECK(beta.dtype() == dtype);

    TORCH_CHECK(gamma.sizes() == beta.sizes());
    TORCH_CHECK(gamma.numel() == cols);

    TORCH_CHECK(epsilon >= 0.f);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto y = torch::empty_like(x);

    auto opts = x.options();

    auto mu = torch::empty({rows}, opts.dtype(torch::kFloat32));
    auto rsigma = torch::empty({rows}, opts.dtype(torch::kFloat32));

    ln_fwd_cuda(y, mu, rsigma, x, gamma, beta, epsilon, rows, cols, stream);

    return {y, mu, rsigma};
}



std::vector<at::Tensor> ln_bwd(const at::Tensor &dw,     // BxSxhidden_size
                               const at::Tensor &x,      // BxSxhidden_size
                               const at::Tensor &mu,     // BxS, FP32!
                               const at::Tensor &rsigma, // BxS, FP32!
                               const at::Tensor &gamma   // hidden_size
) {

  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(dw.is_cuda());
  TORCH_CHECK(mu.is_cuda());
  TORCH_CHECK(rsigma.is_cuda());
  TORCH_CHECK(gamma.is_cuda());

  TORCH_CHECK(x.is_contiguous());
  TORCH_CHECK(dw.is_contiguous());

  auto sizes = x.sizes();
  TORCH_CHECK(sizes.size() == 2);
  TORCH_CHECK(dw.sizes() == sizes);
  auto rows = sizes[0];
  auto cols = sizes[1];
  
  auto dtype = x.scalar_type();
  TORCH_CHECK(dw.dtype() == dtype);
  TORCH_CHECK(gamma.dtype() == dtype);
  TORCH_CHECK(mu.dtype() == torch::kFloat32);
  TORCH_CHECK(rsigma.dtype() == torch::kFloat32);
  TORCH_CHECK(mu.sizes() == rsigma.sizes());
  TORCH_CHECK(mu.numel() == rows);

  TORCH_CHECK(gamma.numel() == cols);


  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto dx = torch::empty_like(x);
  auto dgamma = torch::empty_like(gamma);
  auto dbeta = torch::empty_like(gamma);
  
  ln_bwd_cuda(dx, dgamma, dbeta, dw, x, mu, rsigma, gamma, rows, cols, stream);

  return {dx, dgamma, dbeta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUDA LayerNorm"; // optional module docstring
  m.def("ln_fwd", &ln_fwd, "Run LayerNorm forward kernel");
  m.def("ln_bwd", &ln_bwd, "Run LayerNorm backward kernel");
}
