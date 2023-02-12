#include <iostream>
#include <map>
#include <vector>
#include <chrono>

#include <torch/extension.h>

// The following header file is found in `PYTORCH_HOME`
#include <aten/src/ATen/native/utils/ParamsHash.h>

#if NVFUSER_THIRDPARTY
#include <fusion.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#else
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#endif

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

std::chrono::time_point<std::chrono::steady_clock> t1;
std::chrono::time_point<std::chrono::steady_clock> t2;
std::chrono::time_point<std::chrono::steady_clock> t3;

bool profile() {
    static bool should_profile = std::getenv("APEX_NVFUSER_PROFILE") != nullptr;
    return should_profile;
}

// Make a tensor that is known to be fully contiguous of dimensionality=ndims,
// but unknown sizes
TensorView* makeContigTensor(size_t ndims, DataType dtype = DataType::Float) {
  return TensorViewBuilder()
      .ndims(ndims)
      .dtype(dtype)
      .contiguity(std::vector<bool>(ndims, true))
      .build();
}

struct InstanceNormKey {
  c10::ScalarType input_dtype;//int8_t dtype;
  c10::ScalarType weight_dtype;
  c10::ScalarType mean_dtype;
  size_t dim;
  bool channels_last;
  bool running_mean;
  bool affine;
};

auto get_dtype(c10::ScalarType dtype) {
  auto ret_dtype = DataType::Float;
  if (dtype == c10::ScalarType::Double) {
    ret_dtype = DataType::Double;
  } else if (dtype == c10::ScalarType::Half) {
    ret_dtype = DataType::Half;
  } else if (dtype == c10::ScalarType::BFloat16) {
    ret_dtype = DataType::BFloat16;
  }
  return ret_dtype;
}

// TODO: doesn't support all combinations of dtype e.g., bias, run_var, ..
// bias is assumed to match weight, run_var is assumed to match run_mean
void setKey(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& run_mean, const bool channels_last, InstanceNormKey& key) {
  memset(&key, 0, sizeof(InstanceNormKey));
  key.input_dtype = input.scalar_type();// static_cast<int8_t>(input.scalar_type());
  key.weight_dtype = weight.scalar_type();
  key.mean_dtype = run_mean.scalar_type();
  key.dim = input.sizes().size();
  key.channels_last = channels_last;
  key.running_mean = run_mean.sizes().size() > 0;
  key.affine = weight.sizes().size() ? true : false;
}

std::unordered_map<InstanceNormKey, std::unique_ptr<FusionExecutorCache>, at::native::ParamsHash<InstanceNormKey>, at::native::ParamsEqual<InstanceNormKey> > forward_fusion_cache;
std::unordered_map<InstanceNormKey, std::unique_ptr<FusionExecutorCache>, at::native::ParamsHash<InstanceNormKey>, at::native::ParamsEqual<InstanceNormKey> > backward_fusion_cache;

std::vector<at::Tensor> instance_norm_nvfuser_forward(
  at::Tensor input,
  at::Tensor weight,
  at::Tensor bias,
  at::Tensor run_mean,
  at::Tensor run_var,
  const bool use_input_stats,
  const float momentum,
  const float eps,
  const bool channels_last) {
    if (profile()) {
      t1 = std::chrono::steady_clock::now();
    }
    InstanceNormKey forward_key;
    setKey(input, weight, run_mean, channels_last, forward_key);
    if (forward_fusion_cache.find(forward_key) == forward_fusion_cache.end()) {
      auto fusion = std::make_unique<Fusion>();
      FusionGuard fg(fusion.get());

      const auto _input_dtype = get_dtype(input.scalar_type());
      const auto _weight_dtype = get_dtype(weight.scalar_type());
      const auto _bias_dtype = get_dtype(bias.scalar_type());
      const auto _running_mean_dtype = get_dtype(run_mean.scalar_type());
      const auto _running_var_dtype = get_dtype(run_var.scalar_type());
      auto _input = makeContigTensor(input.sizes().size(), _input_dtype);
      auto _weight = makeContigTensor(weight.sizes().size(), _weight_dtype);
      auto _bias = makeContigTensor(bias.sizes().size(), _bias_dtype);
      auto _running_mean = makeContigTensor(run_mean.sizes().size(), get_dtype(run_mean.scalar_type()));
      auto _running_var = makeContigTensor(run_var.sizes().size(), get_dtype(run_var.scalar_type()));

      fusion->addInput(_input);
      fusion->addInput(_weight);
      fusion->addInput(_bias);

      if (_input_dtype == DataType::Half || _input_dtype == DataType::BFloat16) {
        _input = castOp(DataType::Float, _input);
      }
      if (_weight_dtype == DataType::Half || _weight_dtype == DataType::BFloat16) {
        _weight = castOp(DataType::Float, _weight);
      }
      if (_bias_dtype == DataType::Half || _bias_dtype == DataType::BFloat16) {
        _bias = castOp(DataType::Float, _bias);
      }

      // TODO: decide if passing an empty tensor is the best way to signal no running mean/var
      if (run_mean.sizes().size()) {
        fusion->addInput(_running_mean);
        fusion->addInput(_running_var);
        // casting is done by Forward for running mean/var as it needs original inputs for aliasing
      }

      Double* _momentum = IrBuilder::create<Double>();
      Double* _eps = IrBuilder::create<Double>();
      fusion->addInput(_momentum);
      fusion->addInput(_eps);

      ForwardNormResult result;
      if (!run_mean.sizes().size()) {
        _running_mean = nullptr;
        _running_var = nullptr;
      }
      if (!weight.sizes().size()) {
        _weight = nullptr;
        _bias = nullptr;
      }
      result = instance_norm(
      _input, _weight, _bias, _running_mean, _running_var, use_input_stats, _momentum, _eps, channels_last);

      if (_input_dtype == DataType::Half || _input_dtype == DataType::BFloat16) {
          fusion->addOutput(castOp(_input_dtype, result.output));
          fusion->addOutput(castOp(_input_dtype, result.mean));
          fusion->addOutput(castOp(_input_dtype, result.invstd));
      } else {
          fusion->addOutput(result.output);
          fusion->addOutput(result.mean);
          fusion->addOutput(result.invstd);
      }
      forward_fusion_cache.emplace(forward_key, std::make_unique<FusionExecutorCache>(std::move(fusion)));
    }
    std::vector<torch::jit::IValue> aten_inputs = {input, weight, bias};
    if (run_mean.sizes().size()) {
      aten_inputs.push_back(run_mean);
      aten_inputs.push_back(run_var);
    }
    aten_inputs.push_back(momentum);
    aten_inputs.push_back(eps);
    if (profile()) {
      t2 = std::chrono::steady_clock::now();
    }
    auto r = forward_fusion_cache[forward_key].get()->runFusionWithInputs(aten_inputs);
    if (profile()) {
      t3 = std::chrono::steady_clock::now();
      std::chrono::duration<double> full = t3 - t1;
      std::chrono::duration<double> pre = t2 - t1;
      std::chrono::duration<double> exec = t3 - t2;
      std::cout << "NVFuserInstanceNorm Forward (full, pre-exec, exec) (" << full.count()
                << ", " << pre.count() << ", " << exec.count() << ")" << std::endl;
    }
    return r;
}

std::vector<at::Tensor> instance_norm_nvfuser_backward(
  at::Tensor input,
  at::Tensor grad_output,
  at::Tensor weight,
  at::Tensor run_mean,
  at::Tensor run_var,
  at::Tensor save_mean,
  at::Tensor save_invstd,
  const bool use_input_stats,
  const float eps,
  // const std::vector<bool>& output_mask,
  bool channels_last
  ) {
    if (profile()) {
      t1 = std::chrono::steady_clock::now();
    }
    InstanceNormKey backward_key;
    memset(&backward_key, 0, sizeof(InstanceNormKey));
    setKey(input, weight, run_mean, channels_last, backward_key);
    if (backward_fusion_cache.find(backward_key) == backward_fusion_cache.end()) {
      auto fusion = std::make_unique<Fusion>();
      FusionGuard fg(fusion.get());
      const auto _input_dtype = get_dtype(input.scalar_type());
      const auto _grad_output_dtype = get_dtype(grad_output.scalar_type());
      const auto _weight_dtype = get_dtype(weight.scalar_type());
      const auto _running_mean_dtype = get_dtype(run_mean.scalar_type());
      const auto _running_var_dtype = get_dtype(run_var.scalar_type());
      auto _input = makeContigTensor(input.sizes().size(), _input_dtype);
      auto _grad_output = makeContigTensor(grad_output.sizes().size(), _grad_output_dtype);
      auto _weight = makeContigTensor(weight.sizes().size(), _weight_dtype);
      auto _running_mean = makeContigTensor(run_mean.sizes().size(), get_dtype(run_mean.scalar_type()));
      auto _running_var = makeContigTensor(run_var.sizes().size(), get_dtype(run_var.scalar_type()));
      auto _save_mean = makeContigTensor(save_mean.sizes().size(), get_dtype(save_mean.scalar_type()));
      auto _save_invstd = makeContigTensor(save_invstd.sizes().size(), get_dtype(save_invstd.scalar_type()));

      fusion->addInput(_input);
      fusion->addInput(_grad_output);
      fusion->addInput(_weight);
      fusion->addInput(_running_mean);
      fusion->addInput(_running_var);
      fusion->addInput(_save_mean);
      fusion->addInput(_save_invstd);

      if (_input_dtype == DataType::Half || _input_dtype == DataType::BFloat16) {
        _input = castOp(DataType::Float, _input);
      }
      if (_grad_output_dtype == DataType::Half || _grad_output_dtype == DataType::BFloat16) {
        _grad_output = castOp(DataType::Float, _grad_output);
      }
      if (_weight_dtype == DataType::Half || _weight_dtype == DataType::BFloat16) {
        _weight = castOp(DataType::Float, _weight);
      }
      if (_running_mean_dtype == DataType::Half || _running_mean_dtype == DataType::BFloat16) {
        _running_mean = castOp(DataType::Float, _running_mean);
      }
      if (_running_var_dtype == DataType::Half || _running_var_dtype == DataType::BFloat16) {
        _running_var = castOp(DataType::Float, _running_var);
      }

      Double* _eps = IrBuilder::create<Double>();
      fusion->addInput(_eps);
      if (!run_mean.sizes().size()) {
        _running_mean = nullptr;
        _running_var = nullptr;
      }
      if (!weight.sizes().size()) {
        _weight = nullptr;
      }
      auto result = instance_norm_backward(_input,
                                           _grad_output,
                                           _weight,
                                           _running_mean,
                                           _running_var,
                                           _save_mean,
                                           _save_invstd,
                                           use_input_stats,
                                           _eps,
                                           {true, true, true}, // TODO: is output mask useful?
                                           channels_last);
      if (_input_dtype == DataType::Half || _input_dtype == DataType::BFloat16) {
          fusion->addOutput(castOp(_input_dtype, result.grad_input));
          fusion->addOutput(castOp(_input_dtype, result.grad_weight));
          fusion->addOutput(castOp(_input_dtype, result.grad_bias));
      } else {
          fusion->addOutput(result.grad_input);
          fusion->addOutput(result.grad_weight);
          fusion->addOutput(result.grad_bias);
      }
      backward_fusion_cache.emplace(backward_key, std::make_unique<FusionExecutorCache>(std::move(fusion)));
    }
    std::vector<torch::jit::IValue> aten_inputs = {
      input, grad_output, weight, run_mean, run_var, save_mean, save_invstd, eps};
    if (profile()) {
      t2 = std::chrono::steady_clock::now();
    }
    auto r = backward_fusion_cache[backward_key].get()->runFusionWithInputs(aten_inputs);
    if (profile()) {
      t3 = std::chrono::steady_clock::now();
      std::chrono::duration<double> full = t3 - t1;
      std::chrono::duration<double> pre = t2 - t1;
      std::chrono::duration<double> exec = t3 - t2;
      std::cout << "NVFuserInstanceNorm Backward (full, pre-exec, exec) (" << full.count()
                << ", " << pre.count() << ", " << exec.count() << ")" << std::endl;
    }
    return r;
  }
