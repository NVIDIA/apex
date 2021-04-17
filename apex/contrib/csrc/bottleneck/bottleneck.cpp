#include <ATen/ATen.h>
#include <ATen/cudnn/Handle.h>  // for getcudnnhandle
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <cudnn_frontend.h>

#include <iostream>

#ifdef DEBUG
#define DEBUG_MSG(str) do { std::cout << str << std::endl; } while( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#endif

#ifdef DEBUG_CUDNN
#define DEBUG_CUDNN_MSG(buf, str) do { buf << str << std::endl; } while( false )
#else
#define DEBUG_CUDNN_MSG(buf, str) do { } while ( false )
#endif

#define checkCudnnErr(...)                                                        \
    do {                                                                          \
        int err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        if (err) {                                                                \
            return;                                                    \
	}                                                                         \
    } while (0)


int checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int)code, cudnnGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

void checkError(cudaError_t code, char const * func, const char *file, const int line, bool abort = true);
#define checkCUDAError(val) { checkError((val), #val, __FILE__, __LINE__); }    // in-line regular function

void checkError(cudaError_t code, char const * func, const char *file, const int line, bool abort)
{
  if (code != cudaSuccess)
  {
    const char * errorMessage = cudaGetErrorString(code);
    fprintf(stderr, "CUDA error returned from \"%s\" at %s:%d, Error code: %d (%s)\n", func, file, line, code, errorMessage);
    if (abort){
      cudaDeviceReset();
      exit(code);
    }
  }
}

void generateStrides(const int64_t* dimA, int64_t* strideA, int nbDims, cudnnTensorFormat_t filterFormat) {
    // For INT8x4 and INT8x32 we still compute standard strides here to input
    // into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
    if (filterFormat == CUDNN_TENSOR_NCHW) {
        strideA[nbDims - 1] = 1;
        for (int64_t d = nbDims - 2; d >= 0; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
    } else {
        // Here we assume that the format is CUDNN_TENSOR_NHWC
	strideA[1]          = 1;
        strideA[nbDims - 1] = strideA[1] * dimA[1];
        for (int64_t d = nbDims - 2; d >= 2; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
        strideA[0] = strideA[2] * dimA[2];
    }
}


int getFwdConvDilatedFilterDim(int filterDim, int dilation) {
    return ((filterDim - 1) * dilation) + 1;
}

int getFwdConvPaddedImageDim(int tensorDim, int pad) {
    return tensorDim + (2 * pad);
}

int getFwdConvOutputDim(
    int tensorDim,
    int pad,
    int filterDim,
    int stride,
    int dilation)
{
    int p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1;
    return (p);
}

enum {
    X_TENSOR,
    Y_TENSOR,
    W_TENSOR,
    Z_TENSOR,
    B_TENSOR,
    AFTERADD_TENSOR,
    AFTERBIAS_TENSOR,
    AFTERCONV_TENSOR,
    OPTIONAL,
    AFTEROPT_TENSOR,
};

using common_conv_descriptors =
    std::tuple<cudnn_frontend::Tensor, cudnn_frontend::Tensor, cudnn_frontend::Tensor, cudnn_frontend::ConvDesc>;


common_conv_descriptors
create_common_descriptors(int64_t* x_dim_padded,
                          int64_t* padA,
                          int64_t* convstrideA,
                          int64_t* dilationA,
                          int64_t* w_dim_padded,
                          int64_t* y_dim_padded,
                          cudnnDataType_t dataType,
                          cudnnConvolutionMode_t mode) {
    const int convDim = 2;

    int64_t strideA_padded[4];
    int64_t outstrideA_padded[4];
    int64_t filterstrideA_padded[4];

    generateStrides(w_dim_padded, filterstrideA_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(x_dim_padded, strideA_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(y_dim_padded, outstrideA_padded, 4, CUDNN_TENSOR_NHWC);

    return common_conv_descriptors(cudnn_frontend::TensorBuilder()
                                       .setDim(4, x_dim_padded)
                                       .setStrides(4, strideA_padded)
                                       .setId('x')
                                       .setAlignment(16)
                                       .setDataType(dataType)
                                       .build(),
                                   cudnn_frontend::TensorBuilder()
                                       .setDim(4, y_dim_padded)
                                       .setStrides(4, outstrideA_padded)
                                       .setId('y')
                                       .setAlignment(16)
                                       .setDataType(dataType)
                                       .build(),
                                   cudnn_frontend::TensorBuilder()
                                       .setDim(4, w_dim_padded)
                                       .setStrides(4, filterstrideA_padded)
                                       .setId('w')
                                       .setAlignment(16)
                                       .setDataType(dataType)
                                       .build(),
                                   cudnn_frontend::ConvDescBuilder()
                                       .setDataType(CUDNN_DATA_FLOAT)
                                       .setMathMode(mode)
                                       .setNDims(convDim)
                                       .setStrides(convDim, convstrideA)
                                       .setPrePadding(convDim, padA)
                                       .setPostPadding(convDim, padA)
                                       .setDilation(convDim, dilationA)
                                       .build());
}

using common_convbias_descriptors = std::tuple<cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor>;

common_convbias_descriptors
create_conv_bias_add_act_descriptors(int64_t* x_dim_padded,
                                     int64_t* padA,
                                     int64_t* convstrideA,
                                     int64_t* dilationA,
                                     int64_t* w_dim_padded,
                                     int64_t* y_dim_padded,
                                     cudnnDataType_t dataType) {
    const int convDim = 2;

    int64_t b_dim_padded[4];
    b_dim_padded[0] = 1;
    b_dim_padded[1] = y_dim_padded[1];
    b_dim_padded[2] = 1;
    b_dim_padded[3] = 1;

    int64_t x_stride_padded[4];
    int64_t y_stride_padded[4];
    int64_t w_stride_padded[4];
    int64_t b_stride_padded[4];

    generateStrides(w_dim_padded, w_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(x_dim_padded, x_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(y_dim_padded, y_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(b_dim_padded, b_stride_padded, 4, CUDNN_TENSOR_NHWC);

    return common_convbias_descriptors(cudnn_frontend::TensorBuilder()
                                           .setDim(4, x_dim_padded)
                                           .setStrides(4, x_stride_padded)
                                           .setId('x')
                                           .setAlignment(16)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('y')
                                           .setAlignment(16)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, w_dim_padded)
                                           .setStrides(4, w_stride_padded)
                                           .setId('w')
                                           .setAlignment(16)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, b_dim_padded)
                                           .setStrides(4, b_stride_padded)
                                           .setId('z')
                                           .setAlignment(16)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, b_dim_padded)
                                           .setStrides(4, b_stride_padded)
                                           .setId('b')
                                           .setAlignment(16)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setVirtual()
                                           .setId('A')  // after add
                                           .setAlignment(16)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setVirtual()
                                           .setId('B')  // after bias
                                           .setAlignment(16)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('C')  // after conv
                                           .setAlignment(16)
                                           .setVirtual()
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('i')
                                           .setAlignment(16)
                                           .setDataType(dataType)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('D')  // after optional add
                                           .setAlignment(16)
                                           .setVirtual()
                                           .setDataType(dataType)
                                           .build());
}

// tensor descriptors used for dgrad
enum {
    X_OR_DX_TENSOR,
    DY_TENSOR,
    W_OR_DW_TENSOR,
    SCALE_TENSOR,
    RELU_TENSOR,
    AFTER_DCONV_TENSOR,
    AFTER_DRELU_TENSOR,
};

using dconv_descriptors = std::tuple<cudnn_frontend::Tensor,
                                     cudnn_frontend::Tensor,
                                     cudnn_frontend::Tensor,
                                     cudnn_frontend::Tensor,
                                     cudnn_frontend::Tensor,
                                     cudnn_frontend::Tensor,
                                     cudnn_frontend::Tensor>;

dconv_descriptors
create_dconv_descriptors(int64_t* x_dim_padded,
                         int64_t* padA,
                         int64_t* convstrideA,
                         int64_t* dilationA,
                         int64_t* w_dim_padded,
                         int64_t* y_dim_padded,
                         cudnnDataType_t dataType) {
    const int convDim = 2;

    int64_t b_dim_padded[4];
    b_dim_padded[0] = 1;
    b_dim_padded[1] = x_dim_padded[1];
    b_dim_padded[2] = 1;
    b_dim_padded[3] = 1;

    int64_t x_stride_padded[4];
    int64_t y_stride_padded[4];
    int64_t w_stride_padded[4];
    int64_t b_stride_padded[4];

    generateStrides(w_dim_padded, w_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(x_dim_padded, x_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(y_dim_padded, y_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(b_dim_padded, b_stride_padded, 4, CUDNN_TENSOR_NHWC);

    return dconv_descriptors(cudnn_frontend::TensorBuilder()
                             .setDim(4, x_dim_padded)
                             .setStrides(4, x_stride_padded)
                             .setId('x')
                             .setAlignment(16)
                             .setDataType(dataType)
                             .build(),
                             cudnn_frontend::TensorBuilder()
                             .setDim(4, y_dim_padded)
                             .setStrides(4, y_stride_padded)
                             .setId('y')
                             .setAlignment(16)
                             .setDataType(dataType)
                             .build(),
                             cudnn_frontend::TensorBuilder()
                             .setDim(4, w_dim_padded)
                             .setStrides(4, w_stride_padded)
                             .setId('w')
                             .setAlignment(16)
                             .setDataType(dataType)
                             .build(),
                             cudnn_frontend::TensorBuilder()
                             .setDim(4, b_dim_padded)
                             .setStrides(4, b_stride_padded)
                             .setId('s')
                             .setAlignment(16)
                             .setDataType(dataType)
                             .build(),
                             cudnn_frontend::TensorBuilder()
                             .setDim(4, x_dim_padded)
                             .setStrides(4, x_stride_padded)
                             .setId('r')
                             .setAlignment(16)
                             .setDataType(dataType)
                             .build(),
                             cudnn_frontend::TensorBuilder()
                             .setDim(4, x_dim_padded)
                             .setStrides(4, x_stride_padded)
                             .setVirtual()
                             .setId('A')  // after dconv
                             .setAlignment(16)
                             .setDataType(dataType)
                             .build(),
                             cudnn_frontend::TensorBuilder()
                             .setDim(4, x_dim_padded)
                             .setStrides(4, x_stride_padded)
                             .setVirtual()
                             .setId('B')  // after drelu
                             .setAlignment(16)
                             .setDataType(dataType)
                             .build());
}

// create a cache for plan
std::unordered_map<std::string, cudnn_frontend::ExecutionPlan> plan_cache;

// TODO: better name
std::string getConvFusionString(int64_t* x_dim_padded,
                                int64_t* padA,
                                int64_t* convstrideA,
                                int64_t* dilationA,
                                int64_t* w_dim_padded,
                                cudnnDataType_t dataType,
                                std::string fusion_string) {

  for(int i=0;i<4;i++) {
    fusion_string += 'X';
    fusion_string += std::to_string(x_dim_padded[i]);
  }
  for(int i=0;i<4;i++) {
    fusion_string += 'W';
    fusion_string += std::to_string(w_dim_padded[i]);
  }
  for(int i=0;i<2;i++) {
    fusion_string += 'P';
    fusion_string += std::to_string(padA[i]);
  }
  for(int i=0;i<2;i++) {
    fusion_string += 'S';
    fusion_string += std::to_string(convstrideA[i]);
  }
  for(int i=0;i<2;i++) {
    fusion_string += 'D';
    fusion_string += std::to_string(dilationA[i]);
  }
  fusion_string += 'T';
  fusion_string += std::to_string(dataType);
  return fusion_string;
}

cudnn_frontend::ExecutionPlan& getOrCreatePlan(cudnnHandle_t handle_,
                                               std::stringstream& log_buf,
                                               cudnn_frontend::OperationGraph& opGraph,
                                               std::string cache_string,
                                               bool use_heuristic = true){
  auto it = plan_cache.find(cache_string);
  if (it != plan_cache.end()) {
    DEBUG_CUDNN_MSG(log_buf, "Found plan in cache");
    return it->second;
  } else {
    if (use_heuristic){
      // TODO: confirm which mode to use
      auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
        .setOperationGraph(opGraph)
        .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
        .build();
      // try 3 times for now as WAR for no heuristic training
      int max_tries = 3, count = 0;
      auto& engine_configs = heuristics.getEngineConfig(max_tries);
      while(true) {
        try {
          plan_cache.emplace(cache_string, std::move(cudnn_frontend::ExecutionPlanBuilder()
                                                     .setHandle(handle_)
                                                     .setEngineConfig(engine_configs[count], opGraph.getTag())
                                                     .build()));
          break;
        } catch (cudnn_frontend::cudnnException e) {
          if (++count == max_tries) throw e;
        }
      }
    }else{
    DEBUG_CUDNN_MSG(log_buf, "No plan in cache");
    // How many engines support this operation graph ?
    auto total_engines = opGraph.getEngineCount();
    DEBUG_CUDNN_MSG(log_buf, opGraph.describe() << " has " << total_engines << " engines.");
    // We have to randomly pick one engine from [0, total_engines)
    // Selecting "0" by default
    auto engine = cudnn_frontend::EngineBuilder().setGlobalEngineIdx(0).setOperationGraph(opGraph).build();
    DEBUG_CUDNN_MSG(log_buf, engine.describe());
    auto& knobs = engine.getSupportedKnobs();
    for (auto it = std::begin(knobs); it != std::end(knobs); ++it) {
      DEBUG_CUDNN_MSG(log_buf, it->describe());
    }
    if (knobs.begin() != knobs.end()) {
      DEBUG_CUDNN_MSG(log_buf, "Updated knob choice");
      knobs.begin()->setChoice(knobs.begin()->getMinValue() + 1);
      DEBUG_CUDNN_MSG(log_buf, knobs.begin()->describe());
    }

    // Createmplacee the requisite engine config
    auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
    DEBUG_CUDNN_MSG(log_buf, engine_config.describe());
    plan_cache.emplace(cache_string, std::move(cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(engine_config).build()));
    }

    return plan_cache.find(cache_string)->second;
  }
}

void
run_conv_scale_bias_add_activation(int64_t* x_dim_padded,
                                   int64_t* pad,
                                   int64_t* convstride,
                                   int64_t* dilation,
                                   int64_t* w_dim_padded,
                                   int64_t* y_dim_padded,
                                   cudnnDataType_t dataType,
                                   at::Half* devPtrX,
                                   at::Half* devPtrW,
                                   at::Half* devPtrY,
                                   at::Half* devPtrZ,
                                   at::Half* devPtrB,
                                   at::Half* devPtrI) {
    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;
    try {
        int convDim = 2;

        // Creates the necessary tensor descriptors
        common_convbias_descriptors tensors = create_conv_bias_add_act_descriptors(
            x_dim_padded, pad, convstride, dilation, w_dim_padded, y_dim_padded, dataType);
        DEBUG_CUDNN_MSG(log_buf, std::get<X_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<Y_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<W_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<Z_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<B_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTERADD_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTERBIAS_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTERCONV_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<OPTIONAL>(tensors).describe());

        // Define the add operation
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_MUL)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, scaleDesc.describe());

        // Define the bias operation
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, biasDesc.describe());

        // optional add
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, addDesc.describe());

        // Define the activation operation
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, actDesc.describe());

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, convDesc.describe());

        float alpha  = 1.0f;
        float beta   = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(std::get<X_TENSOR>(tensors))
                           .setwDesc(std::get<W_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERCONV_TENSOR>(tensors))
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, conv_op.describe());

        // Create a Add Node with scaling parameters.
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(std::get<Z_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERADD_TENSOR>(tensors))
                           .setpwDesc(scaleDesc)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, scale_op.describe());

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(scale_op.getOutputTensor())
                           .setbDesc(std::get<B_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERBIAS_TENSOR>(tensors))
                           .setpwDesc(biasDesc)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, bias_op.describe());

        // Create a optional add Node.
        auto add_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(bias_op.getOutputTensor())
                           .setbDesc(std::get<OPTIONAL>(tensors))
                           .setyDesc(std::get<AFTEROPT_TENSOR>(tensors))
                           .setpwDesc(addDesc)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, add_op.describe());


        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setxDesc(devPtrI ? add_op.getOutputTensor() : bias_op.getOutputTensor())
                          .setyDesc(std::get<Y_TENSOR>(tensors))
                          .setpwDesc(actDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, act_op.describe());

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 5> ops = {&conv_op, &scale_op, &bias_op, devPtrI ? &add_op : &act_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
          .setHandle(handle_)
          .setOperationGraph(devPtrI ? ops.size() : 4, ops.data())
          .build();

        // Create string encoding for plan caching
        auto cache_string = getConvFusionString(x_dim_padded, pad, convstride, dilation, w_dim_padded, dataType, opGraph.getTag());
        DEBUG_CUDNN_MSG(log_buf, "[convstring] " << cache_string);

        auto& plan = getOrCreatePlan(handle_, log_buf, opGraph, cache_string);
        DEBUG_CUDNN_MSG(log_buf, "Plan tag: " << plan.getTag());

        auto workspace_size = plan.getWorkspaceSize();
        DEBUG_CUDNN_MSG(log_buf, plan.describe() << " requires workspace " << workspace_size);

        void* workspace_ptr = nullptr;
        auto workspace_tensor = at::empty({(workspace_size+3)/4}, at::TensorOptions(at::kCUDA).dtype(at::kFloat));
        if (workspace_size > 0) {
          workspace_ptr = workspace_tensor.data_ptr<float>();
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrZ, devPtrB, devPtrI};
        int64_t uids[]    = {'x', 'y', 'w', 'z', 'b', 'i'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
          .setDataPointers(devPtrI ? 6 : 5, data_ptrs)
          .setUids(devPtrI ? 6 : 5, uids)
                               .build();
        DEBUG_CUDNN_MSG(log_buf, "variantPack " << variantPack.describe());
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        checkCudnnErr(status);
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");
    } catch (cudnn_frontend::cudnnException e) {
      std::cout << log_buf.str() << "[ERROR] Exception " << e.what() << std::endl;
    }
}

void
run_conv_scale_bias(int64_t* x_dim_padded,
                    int64_t* pad,
                    int64_t* convstride,
                    int64_t* dilation,
                    int64_t* w_dim_padded,
                    int64_t* y_dim_padded,
                    cudnnDataType_t dataType,
                    at::Half* devPtrX,
                    at::Half* devPtrW,
                    at::Half* devPtrY,
                    at::Half* devPtrZ,
                    at::Half* devPtrB) {
    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;
    try {
        int convDim = 2;

        // Creates the necessary tensor descriptors
        common_convbias_descriptors tensors = create_conv_bias_add_act_descriptors(
            x_dim_padded, pad, convstride, dilation, w_dim_padded, y_dim_padded, dataType);
        DEBUG_CUDNN_MSG(log_buf, std::get<X_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<Y_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<W_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<Z_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<B_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTERADD_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTERBIAS_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTERCONV_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<OPTIONAL>(tensors).describe());

        // Define the add operation
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
          .setMode(CUDNN_POINTWISE_MUL)
          .setMathPrecision(CUDNN_DATA_FLOAT)
          .build();
        DEBUG_CUDNN_MSG(log_buf, scaleDesc.describe());

        // Define the bias operation
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, addDesc.describe());

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, convDesc.describe());

        float alpha  = 1.0f;
        float beta   = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(std::get<X_TENSOR>(tensors))
                           .setwDesc(std::get<W_TENSOR>(tensors))
                           .setyDesc(std::get<AFTERCONV_TENSOR>(tensors))
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, conv_op.describe());

        // Create a Add Node with scaling parameters.
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setxDesc(conv_op.getOutputTensor())
          .setbDesc(std::get<Z_TENSOR>(tensors))
          .setyDesc(std::get<AFTERADD_TENSOR>(tensors)) // TODO: change enum to aftermul
          .setpwDesc(scaleDesc)
          .build();
        DEBUG_CUDNN_MSG(log_buf, scale_op.describe());

        // Create a Bias Node.
        auto add_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setxDesc(scale_op.getOutputTensor())
          .setbDesc(std::get<B_TENSOR>(tensors))
          .setyDesc(std::get<Y_TENSOR>(tensors))
          .setpwDesc(addDesc)
          .build();
        DEBUG_CUDNN_MSG(log_buf, add_op.describe());

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 3> ops = {&conv_op, &scale_op, &add_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
          .setHandle(handle_)
          .setOperationGraph(ops.size(), ops.data())
          .build();

        // Create string encoding for plan caching
        auto cache_string = getConvFusionString(x_dim_padded, pad, convstride, dilation, w_dim_padded, dataType, opGraph.getTag());
        DEBUG_CUDNN_MSG(log_buf, "[convstring] " << cache_string);

        auto& plan = getOrCreatePlan(handle_, log_buf, opGraph, cache_string);
        DEBUG_CUDNN_MSG(log_buf, "Plan tag: " << plan.getTag());

        auto workspace_size = plan.getWorkspaceSize();
        DEBUG_CUDNN_MSG(log_buf, plan.describe() << " requires workspace " << workspace_size);

        void* workspace_ptr = nullptr;
        auto workspace_tensor = at::empty({(workspace_size+3)/4}, at::TensorOptions(at::kCUDA).dtype(at::kFloat));
        if (workspace_size > 0) {
          workspace_ptr = workspace_tensor.data_ptr<float>();
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrZ, devPtrB};
        int64_t uids[]    = {'x', 'y', 'w', 'z', 'b'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
          .setDataPointers(5, data_ptrs)
          .setUids(5, uids)
                               .build();
        DEBUG_CUDNN_MSG(log_buf, "variantPack " << variantPack.describe());
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        checkCudnnErr(status);
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");
    } catch (cudnn_frontend::cudnnException e) {
      std::cout << log_buf.str() << "[ERROR] Exception " << e.what() << std::endl;
    }
}


void
run_dconv_drelu_dscale(int64_t* x_dim_padded,
                       int64_t* pad,
                       int64_t* convstride,
                       int64_t* dilation,
                       int64_t* w_dim_padded,
                       int64_t* y_dim_padded,
                       cudnnDataType_t dataType,
                       at::Half* devPtrX,
                       at::Half* devPtrW,
                       at::Half* devPtrY,
                       at::Half* devPtrZ,
                       at::Half* devPtrR) {
    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;
    try {
        int convDim = 2;

        // Creates the necessary tensor descriptors
        dconv_descriptors tensors = create_dconv_descriptors(
            x_dim_padded, pad, convstride, dilation, w_dim_padded, y_dim_padded, dataType);
        DEBUG_CUDNN_MSG(log_buf, std::get<X_OR_DX_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DY_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<W_OR_DW_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<SCALE_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<RELU_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTER_DCONV_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTER_DRELU_TENSOR>(tensors).describe());

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, convDesc.describe());

        // Define the activation backward operation
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
          .setMode(CUDNN_POINTWISE_RELU_BWD)
          .setMathPrecision(CUDNN_DATA_FLOAT)
          .build();
        DEBUG_CUDNN_MSG(log_buf, actDesc.describe());

        // Define the scale backward operation
        auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
          .setMode(CUDNN_POINTWISE_MUL)
          .setMathPrecision(CUDNN_DATA_FLOAT)
          .build();
        DEBUG_CUDNN_MSG(log_buf, scaleDesc.describe());

        float alpha  = 1.0f;
        float beta   = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
          .setdxDesc(std::get<AFTER_DCONV_TENSOR>(tensors))
          .setwDesc(std::get<W_OR_DW_TENSOR>(tensors))
          .setdyDesc(std::get<DY_TENSOR>(tensors))
          .setcDesc(convDesc)
          .setAlpha(alpha)
          .setBeta(beta)
          .build();
        DEBUG_CUDNN_MSG(log_buf, conv_op.describe());

        // TODO: do we need getOutputTensor(), and what it returns in backward case?
        // Create an relu backward Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setdyDesc(std::get<AFTER_DCONV_TENSOR>(tensors))
          .setxDesc(std::get<RELU_TENSOR>(tensors))
          .setdxDesc(std::get<AFTER_DRELU_TENSOR>(tensors))
          .setpwDesc(actDesc)
          .build();
        DEBUG_CUDNN_MSG(log_buf, act_op.describe());

        // Create a Scale Node.
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setxDesc(std::get<AFTER_DRELU_TENSOR>(tensors))
          .setbDesc(std::get<SCALE_TENSOR>(tensors))
          .setyDesc(std::get<X_OR_DX_TENSOR>(tensors))
          .setpwDesc(scaleDesc)
          .build();
        DEBUG_CUDNN_MSG(log_buf, scale_op.describe());

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 3> ops = {&conv_op, &act_op, &scale_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
          .setHandle(handle_)
          .setOperationGraph(ops.size(), ops.data())
          .build();

        // Create string encoding for plan caching
        auto cache_string = getConvFusionString(x_dim_padded, pad, convstride, dilation, w_dim_padded, dataType, opGraph.getTag());
        DEBUG_CUDNN_MSG(log_buf, "[convstring] " << cache_string);

        auto& plan = getOrCreatePlan(handle_, log_buf, opGraph, cache_string);
        DEBUG_CUDNN_MSG(log_buf, "Plan tag: " << plan.getTag());

        auto workspace_size = plan.getWorkspaceSize();
        DEBUG_CUDNN_MSG(log_buf, plan.describe() << " requires workspace " << workspace_size);

        void* workspace_ptr = nullptr;
        auto workspace_tensor = at::empty({(workspace_size+3)/4}, at::TensorOptions(at::kCUDA).dtype(at::kFloat));
        if (workspace_size > 0) {
          workspace_ptr = workspace_tensor.data_ptr<float>();
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrZ, devPtrR};
        int64_t uids[]    = {'x', 'y', 'w', 's', 'r'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
          .setWorkspacePointer(workspace_ptr)
          .setDataPointers(5, data_ptrs)
          .setUids(5, uids)
          .build();
        DEBUG_CUDNN_MSG(log_buf, "variantPack " << variantPack.describe());
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        checkCudnnErr(status);
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");
    } catch (cudnn_frontend::cudnnException e) {
      std::cout << log_buf.str() << "[ERROR] Exception " << e.what() << std::endl;
    }
}

void
run_dconv(int64_t* x_dim_padded,
          int64_t* pad,
          int64_t* convstride,
          int64_t* dilation,
          int64_t* w_dim_padded,
          int64_t* y_dim_padded,
          cudnnDataType_t dataType,
          at::Half* devPtrX,
          at::Half* devPtrW,
          at::Half* devPtrY,
          cudnnBackendDescriptorType_t mode) {
    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;
    try {
        int convDim = 2;

        // Creates the necessary tensor descriptors
        dconv_descriptors tensors = create_dconv_descriptors(
            x_dim_padded, pad, convstride, dilation, w_dim_padded, y_dim_padded, dataType);
        DEBUG_CUDNN_MSG(log_buf, std::get<X_OR_DX_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DY_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<W_OR_DW_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<SCALE_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<RELU_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTER_DCONV_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTER_DRELU_TENSOR>(tensors).describe());

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, convDesc.describe());

        float alpha  = 1.0f;
        float beta   = 0.0f;

        // Create a convolution Node
        // mode should be one of following
        // CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
        // CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR
        auto conv_op_builder = cudnn_frontend::OperationBuilder(mode);
        if (mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
          conv_op_builder.setdxDesc(std::get<X_OR_DX_TENSOR>(tensors))
            .setwDesc(std::get<W_OR_DW_TENSOR>(tensors))
            .setdyDesc(std::get<DY_TENSOR>(tensors))
            .setcDesc(convDesc)
            .setAlpha(alpha)
            .setBeta(beta);
        }
        else {
          conv_op_builder.setxDesc(std::get<X_OR_DX_TENSOR>(tensors))
            .setdwDesc(std::get<W_OR_DW_TENSOR>(tensors))
            .setdyDesc(std::get<DY_TENSOR>(tensors))
            .setcDesc(convDesc)
            .setAlpha(alpha)
            .setBeta(beta);
        }
        auto conv_op = conv_op_builder.build();
        DEBUG_CUDNN_MSG(log_buf, conv_op.describe());

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 1> ops = {&conv_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
          .setHandle(handle_)
          .setOperationGraph(ops.size(), ops.data())
          .build();

        // Create string encoding for plan caching
        auto cache_string = getConvFusionString(x_dim_padded, pad, convstride, dilation, w_dim_padded, dataType, opGraph.getTag());
        DEBUG_CUDNN_MSG(log_buf, "[convstring] " << cache_string);

        auto& plan = getOrCreatePlan(handle_, log_buf, opGraph, cache_string);
        DEBUG_CUDNN_MSG(log_buf, "Plan tag: " << plan.getTag());

        auto workspace_size = plan.getWorkspaceSize();
        DEBUG_CUDNN_MSG(log_buf, plan.describe() << " requires workspace " << workspace_size);

        void* workspace_ptr = nullptr;
        auto workspace_tensor = at::empty({(workspace_size+3)/4}, at::TensorOptions(at::kCUDA).dtype(at::kFloat));
        if (workspace_size > 0) {
          workspace_ptr = workspace_tensor.data_ptr<float>();
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW};
        int64_t uids[]    = {'x', 'y', 'w'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
          .setWorkspacePointer(workspace_ptr)
          .setDataPointers(3, data_ptrs)
          .setUids(3, uids)
          .build();
        DEBUG_CUDNN_MSG(log_buf, "variantPack " << variantPack.describe());
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        checkCudnnErr(status);
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");
    } catch (cudnn_frontend::cudnnException e) {
      std::cout << log_buf.str() << "[ERROR] Exception " << e.what() << std::endl;
    }
}

void
run_dconv_add(int64_t* x_dim_padded,
              int64_t* pad,
              int64_t* convstride,
              int64_t* dilation,
              int64_t* w_dim_padded,
              int64_t* y_dim_padded,
              cudnnDataType_t dataType,
              at::Half* devPtrX,
              at::Half* devPtrW,
              at::Half* devPtrY,
              at::Half* devPtrR) {
    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;
    try {
        int convDim = 2;

        // Creates the necessary tensor descriptors
        dconv_descriptors tensors = create_dconv_descriptors(
            x_dim_padded, pad, convstride, dilation, w_dim_padded, y_dim_padded, dataType);
        DEBUG_CUDNN_MSG(log_buf, std::get<X_OR_DX_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DY_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<W_OR_DW_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<SCALE_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<RELU_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTER_DCONV_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTER_DRELU_TENSOR>(tensors).describe());

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, convstride)
                            .setPrePadding(convDim, pad)
                            .setPostPadding(convDim, pad)
                            .setDilation(convDim, dilation)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, convDesc.describe());

        // Define the add backward operation
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
          .setMode(CUDNN_POINTWISE_ADD)
          .setMathPrecision(CUDNN_DATA_FLOAT)
          .build();
        DEBUG_CUDNN_MSG(log_buf, addDesc.describe());

        float alpha  = 1.0f;
        float beta   = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
          .setdxDesc(std::get<AFTER_DCONV_TENSOR>(tensors))
          .setwDesc(std::get<W_OR_DW_TENSOR>(tensors))
          .setdyDesc(std::get<DY_TENSOR>(tensors))
          .setcDesc(convDesc)
          .setAlpha(alpha)
          .setBeta(beta)
          .build();
        DEBUG_CUDNN_MSG(log_buf, conv_op.describe());

        // TODO: do we need getOutputTensor(), and what it returns in backward case?
        // Create add Node.
        auto add_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setxDesc(std::get<AFTER_DCONV_TENSOR>(tensors))
          .setbDesc(std::get<RELU_TENSOR>(tensors))
          .setyDesc(std::get<X_OR_DX_TENSOR>(tensors))
          .setpwDesc(addDesc)
          .build();
        DEBUG_CUDNN_MSG(log_buf, add_op.describe());

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 2> ops = {&conv_op, &add_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
          .setHandle(handle_)
          .setOperationGraph(ops.size(), ops.data())
          .build();

        // Create string encoding for plan caching
        auto cache_string = getConvFusionString(x_dim_padded, pad, convstride, dilation, w_dim_padded, dataType, opGraph.getTag());
        DEBUG_CUDNN_MSG(log_buf, "[convstring] " << cache_string);

        auto& plan = getOrCreatePlan(handle_, log_buf, opGraph, cache_string);
        DEBUG_CUDNN_MSG(log_buf, "Plan tag: " << plan.getTag());

        auto workspace_size = plan.getWorkspaceSize();
        DEBUG_CUDNN_MSG(log_buf, plan.describe() << " requires workspace " << workspace_size);

        void* workspace_ptr = nullptr;
        auto workspace_tensor = at::empty({(workspace_size+3)/4}, at::TensorOptions(at::kCUDA).dtype(at::kFloat));
        if (workspace_size > 0) {
          workspace_ptr = workspace_tensor.data_ptr<float>();
        }
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrR};
        int64_t uids[]    = {'x', 'y', 'w', 'r'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
          .setWorkspacePointer(workspace_ptr)
          .setDataPointers(4, data_ptrs)
          .setUids(4, uids)
          .build();
        DEBUG_CUDNN_MSG(log_buf, "variantPack " << variantPack.describe());
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        checkCudnnErr(status);
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error");
    } catch (cudnn_frontend::cudnnException e) {
      std::cout << log_buf.str() << "[ERROR] Exception " << e.what() << std::endl;
    }
}


// inputs contains x,w,z,b,(i)
std::vector<at::Tensor> bottleneck_forward(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs) {

  std::cout << std::fixed;
  // create output vector
  std::vector<at::Tensor> outputs;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // setup dimensions
  int64_t dimA[]         = {0, 0, 0, 0};
  int64_t filterdimA1[]  = {0, 0, 0, 0};
  int64_t filterdimA2[]  = {0, 0, 0, 0};
  int64_t filterdimA3[]  = {0, 0, 0, 0};
  int64_t filterdimA4[]  = {0, 0, 0, 0};

  // All dim calculation after this order of n,c,h,w
  int axis[] {0,1,2,3};
  if (explicit_nhwc) {
    axis[0] = 0;
    axis[1] = 3;
    axis[2] = 1;
    axis[3] = 2;
  }
  for (int dim=0;dim<4;dim++) {
    dimA[dim] = inputs[0].size(axis[dim]);
    filterdimA1[dim] = inputs[1].size(axis[dim]);
    filterdimA2[dim] = inputs[2].size(axis[dim]);
    filterdimA3[dim] = inputs[3].size(axis[dim]);
  }
  if (stride_1X1 != 1 || filterdimA3[0] != dimA[1]) {
    for (int dim=0;dim<4;dim++) {
      filterdimA4[dim] = inputs[10].size(axis[dim]);
    }
  }

  // output dim in n,c,h,w used by backend
  int64_t outdimA1[]     = {0, 0, 0, 0}; // Computed Below
  int64_t outdimA2[]     = {0, 0, 0, 0}; // Computed Below
  int64_t outdimA3[]     = {0, 0, 0, 0}; // Computed Below

  // use these fixed value for test run
  int64_t padA[]        = {0, 0};
  int64_t padA1[]        = {1, 1};
  int64_t dilationA[] = {1, 1};
  int64_t convstrideA[] = {1, 1};
  int64_t convstride1X1[] = {stride_1X1, stride_1X1};

  // compute output from pad/stride/dilation
  outdimA1[0] = dimA[0];
  outdimA1[1] = filterdimA1[0];
  for (int dim = 0; dim < 2; dim++) {
    outdimA1[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA1[dim + 2], convstride1X1[dim], dilationA[dim]);
  }

  outdimA2[0] = outdimA1[0];
  outdimA2[1] = filterdimA2[0];
  for (int dim = 0; dim < 2; dim++) {
    outdimA2[dim + 2] = getFwdConvOutputDim(outdimA1[dim + 2], padA1[dim], filterdimA2[dim + 2], convstrideA[dim], dilationA[dim]);
  }

  outdimA3[0] = outdimA2[0];
  outdimA3[1] = filterdimA3[0];
  for (int dim = 0; dim < 2; dim++) {
    outdimA3[dim + 2] = getFwdConvOutputDim(outdimA2[dim + 2], padA[dim], filterdimA3[dim + 2], convstrideA[dim], dilationA[dim]);
  }

  // Create output tensor in the correct shape in pytorch's view
  int64_t outdim1[]     = {0, 0, 0, 0};
  int64_t outdim2[]     = {0, 0, 0, 0};
  int64_t outdim3[]     = {0, 0, 0, 0};
  if (explicit_nhwc) {
    axis[0] = 0;
    axis[1] = 2;
    axis[2] = 3;
    axis[3] = 1;
  }
  for (int dim=0;dim<4;dim++) {
    outdim1[dim] = outdimA1[axis[dim]];
    outdim2[dim] = outdimA2[axis[dim]];
    outdim3[dim] = outdimA3[axis[dim]];
  }

  // run
  at::Half* x = inputs[0].data_ptr<at::Half>();
  at::Half* w = inputs[1].data_ptr<at::Half>();
  at::Half* z = inputs[4].data_ptr<at::Half>();
  at::Half* b = inputs[7].data_ptr<at::Half>();
  auto out1 = at::empty(outdim1, inputs[0].type(), output_format);
  at::Half* y1 = out1.data_ptr<at::Half>();

  run_conv_scale_bias_add_activation(dimA,
                                     padA,
                                     convstride1X1,
                                     dilationA,
                                     filterdimA1,
                                     outdimA1,
                                     CUDNN_DATA_HALF,
                                     x,
                                     w,
                                     y1,
                                     z,
                                     b,
                                     nullptr);

  DEBUG_MSG("[DEBUG] new relu1 : " << out1.to(at::kFloat).sum().item<float>());

  w = inputs[2].data_ptr<at::Half>();
  z = inputs[5].data_ptr<at::Half>();
  b = inputs[8].data_ptr<at::Half>();
  auto out2 = at::empty(outdim2, inputs[0].type(), output_format);
  at::Half* y2 = out2.data_ptr<at::Half>();

  run_conv_scale_bias_add_activation(outdimA1,
                                     padA1,
                                     convstrideA,
                                     dilationA,
                                     filterdimA2,
                                     outdimA2,
                                     CUDNN_DATA_HALF,
                                     y1,
                                     w,
                                     y2,
                                     z,
                                     b,
                                     nullptr);
  DEBUG_MSG("[DEBUG] new relu2 : " << out2.to(at::kFloat).sum().item<float>());

  // create output of conv3
  auto out3 = at::empty(outdim3, inputs[0].type(), output_format);
  at::Half* y3 = out3.data_ptr<at::Half>();

  // create output of conv4 that may exist
  auto identity = at::empty_like(out3);
  at::Half* yi = identity.data_ptr<at::Half>();

  if (stride_1X1 != 1 || filterdimA3[0] != dimA[1]){

    w = inputs[10].data_ptr<at::Half>();
    z = inputs[11].data_ptr<at::Half>();
    b = inputs[12].data_ptr<at::Half>();
    run_conv_scale_bias(dimA,
                        padA,
                        convstride1X1,
                        dilationA,
                        filterdimA4,
                        outdimA3,
                        CUDNN_DATA_HALF,
                        x,
                        w,
                        yi,
                        z,
                        b);
    DEBUG_MSG("[DEBUG] new downsample : " << identity.to(at::kFloat).sum().item<float>());
  }
  else {
    yi = x;
  }

  w = inputs[3].data_ptr<at::Half>();
  z = inputs[6].data_ptr<at::Half>();
  b = inputs[9].data_ptr<at::Half>();

  run_conv_scale_bias_add_activation(outdimA2,
                                     padA,
                                     convstrideA,
                                     dilationA,
                                     filterdimA3,
                                     outdimA3,
                                     CUDNN_DATA_HALF,
                                     y2,
                                     w,
                                     y3,
                                     z,
                                     b,
                                     yi);
  DEBUG_MSG("[DEBUG] new relu3 : " << out3.to(at::kFloat).sum().item<float>());

  outputs.push_back(out1);
  outputs.push_back(out2);
  outputs.push_back(out3);

  return outputs;
}

std::vector<at::Tensor> bottleneck_backward(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs) {

  bool requires_grad = inputs[0].requires_grad();

  std::cout << std::fixed;
  // create output vector
  std::vector<at::Tensor> outputs;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // setup dimensions
  int64_t dimA[]         = {0, 0, 0, 0};
  int64_t filterdimA1[]  = {0, 0, 0, 0};
  int64_t filterdimA2[]  = {0, 0, 0, 0};
  int64_t filterdimA3[]  = {0, 0, 0, 0};
  int64_t filterdimA4[]  = {0, 0, 0, 0};

  // All dim calculation after this order of n,c,h,w
  int axis[] {0,1,2,3};
  if (explicit_nhwc) {
    axis[0] = 0;
    axis[1] = 3;
    axis[2] = 1;
    axis[3] = 2;
  }
  for (int dim=0;dim<4;dim++) {
    dimA[dim] = inputs[0].size(axis[dim]);
    filterdimA1[dim] = inputs[1].size(axis[dim]);
    filterdimA2[dim] = inputs[2].size(axis[dim]);
    filterdimA3[dim] = inputs[3].size(axis[dim]);
  }
  if (stride_1X1 != 1 || filterdimA3[0] != dimA[1]) {
    for (int dim=0;dim<4;dim++) {
      filterdimA4[dim] = inputs[14].size(axis[dim]);
    }
  }

  // output dim in n,c,h,w used by backend
  int64_t outdimA1[]     = {0, 0, 0, 0}; // Computed Below
  int64_t outdimA2[]     = {0, 0, 0, 0}; // Computed Below
  int64_t outdimA3[]     = {0, 0, 0, 0}; // Computed Below

  // use these fixed value for test run
  int64_t padA[]        = {0, 0};
  int64_t padA1[]        = {1, 1};
  int64_t dilationA[] = {1, 1};
  int64_t convstrideA[] = {1, 1};
  int64_t convstride1X1[] = {stride_1X1, stride_1X1};

  // compute output from pad/stride/dilation
  outdimA1[0] = dimA[0];
  outdimA1[1] = filterdimA1[0];
  for (int dim = 0; dim < 2; dim++) {
    outdimA1[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA1[dim + 2], convstride1X1[dim], dilationA[dim]);
  }

  outdimA2[0] = outdimA1[0];
  outdimA2[1] = filterdimA2[0];
  for (int dim = 0; dim < 2; dim++) {
    outdimA2[dim + 2] = getFwdConvOutputDim(outdimA1[dim + 2], padA1[dim], filterdimA2[dim + 2], convstrideA[dim], dilationA[dim]);
  }

  outdimA3[0] = outdimA2[0];
  outdimA3[1] = filterdimA3[0];
  for (int dim = 0; dim < 2; dim++) {
    outdimA3[dim + 2] = getFwdConvOutputDim(outdimA2[dim + 2], padA[dim], filterdimA3[dim + 2], convstrideA[dim], dilationA[dim]);
  }

  // Create output tensor in the correct shape in pytorch's view
  int64_t outdim1[]     = {0, 0, 0, 0};
  int64_t outdim2[]     = {0, 0, 0, 0};
  int64_t outdim3[]     = {0, 0, 0, 0};
  if (explicit_nhwc) {
    axis[0] = 0;
    axis[1] = 2;
    axis[2] = 3;
    axis[3] = 1;
  }
  for (int dim=0;dim<4;dim++) {
    outdim1[dim] = outdimA1[axis[dim]];
    outdim2[dim] = outdimA2[axis[dim]];
    outdim3[dim] = outdimA3[axis[dim]];
  }

  // dconv3+drelu2+dscale2
  at::Half* conv_in = inputs[13].data_ptr<at::Half>();
  at::Half* dy3 = inputs[10].data_ptr<at::Half>();

  DEBUG_MSG("[DEBUG] new dconv3 : " << inputs[10].to(at::kFloat).sum().item<float>());

  // wgrad
  auto wgrad3 = at::empty_like(inputs[3]);
  at::Half* dw3 = wgrad3.data_ptr<at::Half>();
  run_dconv(outdimA2,
            padA,
            convstrideA,
            dilationA,
            filterdimA3,
            outdimA3,
            CUDNN_DATA_HALF,
            conv_in,
            dw3,
            dy3,
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);

  // dgrad
  auto grad_out2 = at::empty(outdim2, inputs[0].type(), output_format);
  at::Half* dy2 = grad_out2.data_ptr<at::Half>();
  at::Half* w = inputs[3].data_ptr<at::Half>();
  at::Half* z = inputs[5].data_ptr<at::Half>();

  at::Half* relu2 = inputs[13].data_ptr<at::Half>();

  run_dconv_drelu_dscale(outdimA2,
                         padA,
                         convstrideA,
                         dilationA,
                         filterdimA3,
                         outdimA3,
                         CUDNN_DATA_HALF,
                         dy2,
                         w,
                         dy3,
                         z,
                         relu2);

  DEBUG_MSG("[DEBUG] new dconv2 : " << grad_out2.to(at::kFloat).sum().item<float>());

  // dconv2+drelu1+dscale1
  conv_in = inputs[12].data_ptr<at::Half>();

  // wgrad
  auto wgrad2 = at::empty_like(inputs[2]);
  at::Half* dw2 = wgrad2.data_ptr<at::Half>();
  run_dconv(outdimA1,
            padA1,
            convstrideA,
            dilationA,
            filterdimA2,
            outdimA2,
            CUDNN_DATA_HALF,
            conv_in,
            dw2,
            dy2,
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);

  // dgrad
  auto grad_out1 = at::empty(outdim1, inputs[0].type(), output_format);
  at::Half* dy1 = grad_out1.data_ptr<at::Half>();
  w = inputs[2].data_ptr<at::Half>();
  z = inputs[4].data_ptr<at::Half>();

  at::Half* relu1 = inputs[12].data_ptr<at::Half>();
  // fused dgrad
  run_dconv_drelu_dscale(outdimA1,
                         padA1,
                         convstrideA,
                         dilationA,
                         filterdimA2,
                         outdimA2,
                         CUDNN_DATA_HALF,
                         dy1,
                         w,
                         dy2,
                         z,
                         relu1);

/*
  // backward strided conv cannot be fused
  // if stride == 1 but channel changes, we can fuse here
  if (stride_1X1 != 1){
    // dgrad
    run_dconv(outdimA1,
              padA1,
              convstride1X1,
              dilationA,
              filterdimA2,
              outdimA2,
              CUDNN_DATA_HALF,
              dy1,
              w,
              dy2,
              CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);

    // mul fused mask
    grad_out1.mul_(inputs[15]);
  }
  else {
    at::Half* relu1 = inputs[12].data_ptr<at::Half>();
    // fused dgrad
    run_dconv_drelu_dscale(outdimA1,
                           padA1,
                           convstride1X1,
                           dilationA,
                           filterdimA2,
                           outdimA2,
                           CUDNN_DATA_HALF,
                           dy1,
                           w,
                           dy2,
                           z,
                           relu1);
  }
*/
  DEBUG_MSG("[DEBUG] new dconv1 : " << grad_out1.to(at::kFloat).sum().item<float>());

  // create grads of conv4 that may exist
  auto grad_x_conv4 = at::empty_like(inputs[0]);
  at::Half* dx_conv4 = grad_x_conv4.data_ptr<at::Half>();
  at::Tensor wgrad4;

  // x used for dconv1 and dconv4 wgrad
  at::Half* x = inputs[0].data_ptr<at::Half>();

  if (stride_1X1 != 1 || filterdimA3[0] != dimA[1]){
    w = inputs[14].data_ptr<at::Half>();
    at::Half* dy_conv4 = inputs[11].data_ptr<at::Half>();
    if (requires_grad) {
      run_dconv(dimA,
                padA,
                convstride1X1,
                dilationA,
                filterdimA4,
                outdimA3,
                CUDNN_DATA_HALF,
                dx_conv4,
                w,
                dy_conv4,
                CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
      // we don't print here since we can't hook out this grad in pytorch alone to compare, due to addition with dx
      // DEBUG_MSG("[DEBUG] new dx_identity : " << grad_x_conv4.to(at::kFloat).sum().item<float>());
    }
    // wgrad
    wgrad4 = at::empty_like(inputs[14]);
    at::Half* dw4 = wgrad4.data_ptr<at::Half>();
    run_dconv(dimA,
              padA,
              convstride1X1,
              dilationA,
              filterdimA4,
              outdimA3,
              CUDNN_DATA_HALF,
              x,
              dw4,
              dy_conv4,
              CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
  }
  else {
    // if there is no downsample, dx_conv4 is fork of drelu3
    dx_conv4 = inputs[11].data_ptr<at::Half>();
  }

  // dconv1+add
  // wgrad
  auto wgrad1 = at::empty_like(inputs[1]);
  at::Half* dw1 = wgrad1.data_ptr<at::Half>();
  run_dconv(dimA,
            padA,
            convstride1X1,
            dilationA,
            filterdimA1,
            outdimA1,
            CUDNN_DATA_HALF,
            x,
            dw1,
            dy1,
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);

  // dgrad
  w = inputs[1].data_ptr<at::Half>();
  auto grad_x = at::empty_like(inputs[0]);
  at::Half* dx = grad_x.data_ptr<at::Half>();

  // backward strided conv cannot be fused
  // if stride == 1 but channel changes, we can fuse here
  if (requires_grad){
    if (stride_1X1 != 1){
      run_dconv(dimA,
                padA,
                convstride1X1,
                dilationA,
                filterdimA1,
                outdimA1,
                CUDNN_DATA_HALF,
                dx,
                w,
                dy1,
                CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
      // add 2 together
      grad_x.add_(grad_x_conv4);
    }
    else {
      run_dconv_add(dimA,
                    padA,
                    convstride1X1,
                    dilationA,
                    filterdimA1,
                    outdimA1,
                    CUDNN_DATA_HALF,
                    dx,
                    w,
                    dy1,
                    dx_conv4);
    }
  }

  DEBUG_MSG("[DEBUG] new dx : " << grad_x.to(at::kFloat).sum().item<float>());
  DEBUG_MSG("[DEBUG] new wgrad1 : " << wgrad1.to(at::kFloat).sum().item<float>());
  DEBUG_MSG("[DEBUG] new wgrad2 : " << wgrad2.to(at::kFloat).sum().item<float>());
  DEBUG_MSG("[DEBUG] new wgrad3 : " << wgrad3.to(at::kFloat).sum().item<float>());
  outputs.push_back(grad_x);
  outputs.push_back(wgrad1);
  outputs.push_back(wgrad2);
  outputs.push_back(wgrad3);

  if (stride_1X1 != 1 || filterdimA3[0] != dimA[1]) {
    DEBUG_MSG("[DEBUG] new wgrad4 : " << wgrad4.to(at::kFloat).sum().item<float>());
    outputs.push_back(wgrad4);
  }

  return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bottleneck_forward, "Bottleneck block forward");
  m.def("backward", &bottleneck_backward, "Bottleneck block backward");
}
