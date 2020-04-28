#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

#include <stdio.h>

size_t get_mlp_reserved_space(int batch_size, int num_layers, const int* output_features);

template <typename T>
size_t get_mlp_bp_workspace_in_bytes(int batch_size, int num_layers, const int* output_features);

template <typename T>
int mlp_fp(
    T* X,
    int input_features,
    int batch_size,
    T** WPtr,
    int num_layers,
    int* output_features,
    T** BPtr,
    T* Y,
    T* reserved_space,
    int use_bias,
    int activation);

template <typename T>
int mlp_bp(
    T* X,
    T* Y,
    int input_features,
    int batch_size,
    T** WPtr,
    int num_layers,
    int* output_features,
    T* dY,
    T* reserved_space,
    T* work_space,
    T* dX,
    T** dwPtr,
    T** dbPtr,
    bool requires_grad,
    int use_bias,
    int activation);

std::vector<at::Tensor> mlp_forward(int use_bias, int activation, std::vector<at::Tensor> inputs) {

  auto num_layers = inputs.size() - 1;
  if (use_bias) {
    // inputs contains (input, weights, biases)
    num_layers /= 2;
  }
  auto batch_size = inputs[0].size(0);
  auto input_features = inputs[0].size(1);

  std::vector<int> output_features;
  for (int i = 0; i < num_layers; i++) {
    output_features.push_back(inputs[i + 1].size(0));
  }

  auto reserved_size = get_mlp_reserved_space(batch_size, num_layers, output_features.data());

  // create output/workspace tensor
  // TODO(deyuf): just get buffer?
  auto out = at::empty({batch_size, output_features.back()}, inputs[0].type());
  auto reserved_space = at::empty({reserved_size}, inputs[0].type());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs[0].type(), "mlp_forward", [&] {
    std::vector<scalar_t*> w_ptr;
    std::vector<scalar_t*> b_ptr;
    for (int i = 0; i < num_layers; i++) {
      w_ptr.push_back(inputs[i + 1].data_ptr<scalar_t>());
      if (use_bias) {
        b_ptr.push_back(inputs[i + 1 + num_layers].data_ptr<scalar_t>());
      }
    }
    auto result = mlp_fp<scalar_t>(
        inputs[0].data_ptr<scalar_t>(),
        input_features,
        batch_size,
        w_ptr.data(),
        num_layers,
        output_features.data(),
        b_ptr.data(),
        out.data_ptr<scalar_t>(),
        reserved_space.data_ptr<scalar_t>(),
        use_bias,
        activation);
  });

  return {out, reserved_space};
}

std::vector<at::Tensor> mlp_backward(
  int use_bias,
  int activation,
  at::Tensor grad_o,
  std::vector<at::Tensor> fprop_outputs,
  std::vector<at::Tensor> inputs) {

  auto num_layers = inputs.size() - 1;
  if (use_bias) {
    // inputs contains (input, weights, biases)
    num_layers /= 2;
  }

  auto batch_size = inputs[0].size(0);
  auto input_features = inputs[0].size(1);

  // TODO: not creating empty tensor for it?
  bool requires_grad = inputs[0].requires_grad();

  std::vector<int> output_features;
  for (int i = 0; i < num_layers; i++) {
    output_features.push_back(inputs[i + 1].size(0));
  }
  // create outputs, length of inputs
  // TODO: not create bias if not needed
  std::vector<at::Tensor> outputs;
  for (int i = 0; i < inputs.size(); i++) {
    outputs.push_back(at::empty(inputs[i].sizes(), inputs[i].type()));  // clone for testing now
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(inputs[0].type(), "mlp_backward", [&] {
    std::vector<scalar_t*> w_ptr;
    for (int i = 0; i < num_layers; i++) {
      w_ptr.push_back(inputs[i + 1].data_ptr<scalar_t>());
    }
    std::vector<scalar_t*> outputs_ptr;
    for (int i = 0; i < inputs.size(); i++) {
      outputs_ptr.push_back(outputs[i].data_ptr<scalar_t>());
    }

    auto work_size =
        get_mlp_bp_workspace_in_bytes<scalar_t>(batch_size, num_layers, output_features.data());

    // auto work_space = at::empty({work_size*4}, at::kByte);
    auto work_space = at::empty({work_size / sizeof(scalar_t)}, inputs[0].type());

    auto result = mlp_bp<scalar_t>(
        inputs[0].data_ptr<scalar_t>(),
        fprop_outputs[0].data_ptr<scalar_t>(),
        input_features,
        batch_size,
        w_ptr.data(),
        num_layers,
        output_features.data(),
        grad_o.contiguous().data_ptr<scalar_t>(),
        fprop_outputs[1].data_ptr<scalar_t>(),
        work_space.data_ptr<scalar_t>(),
        outputs_ptr[0],
        outputs_ptr.data() + 1,
        outputs_ptr.data() + 1 + num_layers,
        requires_grad,
        use_bias,
        activation);
  });

  return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mlp_forward, "MLP forward");
  m.def("backward", &mlp_backward, "MLP backward");
}
