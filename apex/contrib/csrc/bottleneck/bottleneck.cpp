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
                                           .setDataType(CUDNN_DATA_FLOAT)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setVirtual()
                                           .setId('B')  // after bias
                                           .setAlignment(16)
                                           .setDataType(CUDNN_DATA_FLOAT)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('C')  // after conv
                                           .setAlignment(16)
                                           .setVirtual()
                                           .setDataType(CUDNN_DATA_FLOAT)
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
                                           .setDataType(CUDNN_DATA_FLOAT)
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
                             .setDataType(CUDNN_DATA_FLOAT)
                             .build(),
                             cudnn_frontend::TensorBuilder()
                             .setDim(4, x_dim_padded)
                             .setStrides(4, x_stride_padded)
                             .setVirtual()
                             .setId('B')  // after drelu
                             .setAlignment(16)
                             .setDataType(CUDNN_DATA_FLOAT)
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
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
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
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
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
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
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
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
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
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
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

namespace {

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
    AFTERACT_TENSOR,
    GEN_INDEX_TENSOR,
    MASK_TOP_TENSOR,
    MASK_BOTTOM_TENSOR,
    MASK_TENSOR,
    THRESHOLD_TOP_TENSOR,
    THRESHOLD_BOTTOM_TENSOR,
};

using masked_convbias_descriptors = std::tuple<cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor,
                                               cudnn_frontend::Tensor>;

masked_convbias_descriptors
create_conv_bias_add_act_mask_descriptors(int64_t* x_dim_padded,
                                          int64_t* padA,
					  int64_t* convstrideA,
					  int64_t* dilationA,
					  int64_t* w_dim_padded,
					  int64_t* y_dim_padded,
					  int64_t* threshold_dim,
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
    int64_t threshold_stride[4];

    generateStrides(w_dim_padded, w_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(x_dim_padded, x_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(y_dim_padded, y_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(b_dim_padded, b_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(threshold_dim, threshold_stride, 4, CUDNN_TENSOR_NHWC);

    return masked_convbias_descriptors(cudnn_frontend::TensorBuilder()
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
                                           .setDataType(CUDNN_DATA_FLOAT)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setVirtual()
                                           .setId('B')  // after bias
                                           .setAlignment(16)
                                           .setDataType(CUDNN_DATA_FLOAT)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('C')  // after conv
                                           .setAlignment(16)
                                           .setVirtual()
                                           .setDataType(CUDNN_DATA_FLOAT)
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
                                           .setDataType(CUDNN_DATA_FLOAT)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('E')  // after act for masked
                                           .setAlignment(16)
                                           .setVirtual()
                                           .setDataType(CUDNN_DATA_FLOAT)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('I')  // output of the gen index operation
                                           .setAlignment(16)
                                           .setVirtual()
                                           .setDataType(CUDNN_DATA_INT32)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('m')  // top half of the mask created after the less than
                                           .setAlignment(16)
                                           .setVirtual()
                                           .setDataType(CUDNN_DATA_BOOLEAN)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('n')  // bottom half of the mask
                                           .setAlignment(16)
                                           .setVirtual()
                                           .setDataType(CUDNN_DATA_BOOLEAN)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, y_dim_padded)
                                           .setStrides(4, y_stride_padded)
                                           .setId('M')  // OR of the top and bottom masks
                                           .setAlignment(16)
                                           .setVirtual()
                                           .setDataType(CUDNN_DATA_BOOLEAN)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, threshold_dim)
                                           .setStrides(4, threshold_stride)
                                           .setId('t')  // threshold for creating the top mask
                                           .setAlignment(16)
                                           .setDataType(CUDNN_DATA_INT32)
                                           .build(),
                                       cudnn_frontend::TensorBuilder()
                                           .setDim(4, threshold_dim)
                                           .setStrides(4, threshold_stride)
                                           .setId('u')  // threshold for creating the bottom mask
                                           .setAlignment(16)
                                           .setDataType(CUDNN_DATA_INT32)
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
    DGRAD_INPUT_TENSOR,
    DGRAD_OPTIONAL_TENSOR,
    DGRAD_GEN_INDEX_TENSOR,
    DGRAD_MASK_TOP_TENSOR,
    DGRAD_MASK_BOTTOM_TENSOR,
    DGRAD_MASK_TENSOR,
    DGRAD_THRESHOLD_TOP_TENSOR,
    DGRAD_THRESHOLD_BOTTOM_TENSOR,
};

using dconv_add_descriptors = std::tuple<cudnn_frontend::Tensor,
                                         cudnn_frontend::Tensor,
                                         cudnn_frontend::Tensor,
                                         cudnn_frontend::Tensor,
                                         cudnn_frontend::Tensor,
                                         cudnn_frontend::Tensor,
                                         cudnn_frontend::Tensor,
                                         cudnn_frontend::Tensor,
                                         cudnn_frontend::Tensor>;

dconv_add_descriptors
create_dconv_add_descriptors(int64_t* x_dim_padded,
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

    return dconv_add_descriptors(cudnn_frontend::TensorBuilder()
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
                             .setDataType(CUDNN_DATA_FLOAT)
                             .build(),
                             cudnn_frontend::TensorBuilder()
                             .setDim(4, x_dim_padded)
                             .setStrides(4, x_stride_padded)
                             .setVirtual()
                             .setId('B')  // after drelu
                             .setAlignment(16)
                             .setDataType(CUDNN_DATA_FLOAT)
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
			     .setDataType(CUDNN_DATA_FLOAT)
			     .build());
}

using dconv_mask_descriptors = std::tuple<cudnn_frontend::Tensor,
                                          cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor,
					  cudnn_frontend::Tensor>;

dconv_mask_descriptors
create_dconv_mask_descriptors(int64_t* x_dim_padded,
                              int64_t* padA,
			      int64_t* convstrideA,
			      int64_t* dilationA,
			      int64_t* w_dim_padded,
			      int64_t* y_dim_padded,
			      int64_t* threshold_dim,
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
    int64_t threshold_stride[4];

    generateStrides(w_dim_padded, w_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(x_dim_padded, x_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(y_dim_padded, y_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(b_dim_padded, b_stride_padded, 4, CUDNN_TENSOR_NHWC);
    generateStrides(threshold_dim, threshold_stride, 4, CUDNN_TENSOR_NHWC);

    return dconv_mask_descriptors(cudnn_frontend::TensorBuilder()
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
                             .setDataType(CUDNN_DATA_FLOAT)
                             .build(),
                             cudnn_frontend::TensorBuilder()
                             .setDim(4, x_dim_padded)
                             .setStrides(4, x_stride_padded)
                             .setVirtual()
                             .setId('B')  // after drelu
                             .setAlignment(16)
                             .setDataType(CUDNN_DATA_FLOAT)
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
			     .setDataType(CUDNN_DATA_FLOAT)
			     .build(),
			     cudnn_frontend::TensorBuilder()
			     .setDim(4, y_dim_padded)
			     .setStrides(4, y_stride_padded)
			     .setId('I')  // output of the gen index operation
			     .setAlignment(16)
			     .setVirtual()
			     .setDataType(CUDNN_DATA_INT32)
			     .build(),
			     cudnn_frontend::TensorBuilder()
			     .setDim(4, y_dim_padded)
			     .setStrides(4, y_stride_padded)
			     .setId('m')  // top half of the mask created after the less than
			     .setAlignment(16)
			     .setVirtual()
			     .setDataType(CUDNN_DATA_BOOLEAN)
			     .build(),
			     cudnn_frontend::TensorBuilder()
			     .setDim(4, y_dim_padded)
			     .setStrides(4, y_stride_padded)
			     .setId('n')  // bottom half of the mask
			     .setAlignment(16)
			     .setVirtual()
			     .setDataType(CUDNN_DATA_BOOLEAN)
			     .build(),
			     cudnn_frontend::TensorBuilder()
			     .setDim(4, y_dim_padded)
			     .setStrides(4, y_stride_padded)
			     .setId('M')  // OR of the top and bottom masks
			     .setAlignment(16)
			     .setVirtual()
			     .setDataType(CUDNN_DATA_BOOLEAN)
			     .build(),
			     cudnn_frontend::TensorBuilder()
			     .setDim(4, threshold_dim)
			     .setStrides(4, threshold_stride)
			     .setId('t')  // threshold for creating the top mask
			     .setAlignment(16)
			     .setDataType(CUDNN_DATA_INT32)
			     .build(),
			     cudnn_frontend::TensorBuilder()
			     .setDim(4, threshold_dim)
			     .setStrides(4, threshold_stride)
			     .setId('u')  // threshold for creating the bottom mask
			     .setAlignment(16)
			     .setDataType(CUDNN_DATA_INT32)
			     .build());
}

void
run_conv_add_scale_bias_activation(int64_t* x_dim_padded,
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
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTEROPT_TENSOR>(tensors).describe());

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

        // create an add node.
        auto add_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(std::get<OPTIONAL>(tensors))
                           .setyDesc(std::get<AFTEROPT_TENSOR>(tensors))
                           .setpwDesc(addDesc)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, add_op.describe());

        // Create a Add Node with scaling parameters.
        auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(add_op.getOutputTensor())
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

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(bias_op.getOutputTensor())
                           .setyDesc(std::get<Y_TENSOR>(tensors))
                           .setpwDesc(actDesc)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, act_op.describe());

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 5> ops = {&conv_op, &add_op, &scale_op, &bias_op, &act_op};

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
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrZ, devPtrB, devPtrI};
        int64_t uids[]    = {'x', 'y', 'w', 'z', 'b', 'i'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(6, data_ptrs)
                               .setUids(6, uids)
                               .build();
        DEBUG_CUDNN_MSG(log_buf, "variantPack " << variantPack.describe());
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        checkCudnnErr(status);
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
    } catch (cudnn_frontend::cudnnException e) {
      std::cout << log_buf.str() << "[ERROR] Exception " << e.what() << std::endl;
    }
}

void
run_conv_scale_bias_add_activation_mask(int64_t* x_dim_padded,
                                        int64_t* pad,
                                        int64_t* convstride,
					int64_t* dilation,
					int64_t* w_dim_padded,
					int64_t* y_dim_padded,
					int64_t* threshold_dim,
					cudnnDataType_t dataType,
					at::Half* devPtrX,
					at::Half* devPtrW,
					at::Half* devPtrY,
					at::Half* devPtrZ,
					at::Half* devPtrB,
					at::Half* devPtrI,
					int* devPtrT,
					int* devPtrU,
					int axis) {
    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;
    try {
        int convDim = 2;

        // Creates the necessary tensor descriptors
        masked_convbias_descriptors tensors = create_conv_bias_add_act_mask_descriptors(
            x_dim_padded, pad, convstride, dilation, w_dim_padded, y_dim_padded, threshold_dim, dataType);
        DEBUG_CUDNN_MSG(log_buf, std::get<X_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<Y_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<W_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<Z_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<B_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTERADD_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTERBIAS_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTERCONV_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<OPTIONAL>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTERACT_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<GEN_INDEX_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<MASK_TOP_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<MASK_BOTTOM_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<MASK_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<THRESHOLD_TOP_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<THRESHOLD_BOTTOM_TENSOR>(tensors).describe());

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

        // Define the genIndex descriptor
        auto genIndexDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_GEN_INDEX)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .setAxis(axis)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, genIndexDesc.describe());

        // Define the lessThan descriptor
        auto lessThanDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_CMP_LT)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, lessThanDesc.describe());

        // Define the greaterThan descriptor
        auto greaterThanDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_CMP_GT)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, greaterThanDesc.describe());

        // Define the logical_or descriptor
        auto logicalOrDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_LOGICAL_OR)
                           .setMathPrecision(CUDNN_DATA_BOOLEAN)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, logicalOrDesc.describe());

        // Define the binary_selection descriptor
        auto selectionDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_BINARY_SELECT)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, selectionDesc.describe());

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
                          .setyDesc(std::get<AFTERACT_TENSOR>(tensors))
                          .setpwDesc(actDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, act_op.describe());

        // Create a Gen_Index Node.
        auto genIndex_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(std::get<AFTERACT_TENSOR>(tensors))
                          .setyDesc(std::get<GEN_INDEX_TENSOR>(tensors))
                          .setpwDesc(genIndexDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, genIndex_op.describe());

        // Create a LessThan Node.
        auto lessThan_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(std::get<GEN_INDEX_TENSOR>(tensors))
                          .setbDesc(std::get<THRESHOLD_TOP_TENSOR>(tensors))
                          .setyDesc(std::get<MASK_TOP_TENSOR>(tensors))
                          .setpwDesc(lessThanDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, lessThan_op.describe());

        // Create a GreaterThan Node.
        auto greaterThan_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(std::get<GEN_INDEX_TENSOR>(tensors))
                          .setbDesc(std::get<THRESHOLD_BOTTOM_TENSOR>(tensors))
                          .setyDesc(std::get<MASK_BOTTOM_TENSOR>(tensors))
                          .setpwDesc(greaterThanDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, greaterThan_op.describe());

        // Create a LogicalOr Node.
        auto logicalOr_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(std::get<MASK_TOP_TENSOR>(tensors))
                          .setbDesc(std::get<MASK_BOTTOM_TENSOR>(tensors))
                          .setyDesc(std::get<MASK_TENSOR>(tensors))
                          .setpwDesc(logicalOrDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, logicalOr_op.describe());

        // Create a Binary_Selection Node.
        auto selection_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(std::get<AFTERCONV_TENSOR>(tensors))
                          .setbDesc(std::get<AFTERACT_TENSOR>(tensors))
                          .settDesc(std::get<MASK_TENSOR>(tensors))
                          .setyDesc(std::get<Y_TENSOR>(tensors))
                          .setpwDesc(selectionDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, selection_op.describe());

        // Create an Operation Graph. In this case it is convolution add bias activation
	if (devPtrI) {

	  std::array<cudnn_frontend::Operation const*, 10> ops = {&conv_op, &scale_op, &bias_op, &add_op, &act_op, &genIndex_op, &lessThan_op, &greaterThan_op, &logicalOr_op, &selection_op};

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
	  void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrZ, devPtrB, devPtrI, devPtrT, devPtrU};
	  int64_t uids[]    = {'x', 'y', 'w', 'z', 'b', 'i', 't', 'u'};
	  auto variantPack  = cudnn_frontend::VariantPackBuilder()
	    .setWorkspacePointer(workspace_ptr)
	    .setDataPointers(8, data_ptrs)
	    .setUids(8, uids)
	    .build();
	  DEBUG_CUDNN_MSG(log_buf, "variantPack " << variantPack.describe());
	  cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
	  checkCudnnErr(status);
	  cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
	} else {

	  std::array<cudnn_frontend::Operation const*, 9> ops = {&conv_op, &scale_op, &bias_op, &act_op, &genIndex_op, &lessThan_op, &greaterThan_op, &logicalOr_op, &selection_op};

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
	  void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrZ, devPtrB, devPtrT, devPtrU};
	  int64_t uids[]    = {'x', 'y', 'w', 'z', 'b', 't', 'u'};
	  auto variantPack  = cudnn_frontend::VariantPackBuilder()
	    .setWorkspacePointer(workspace_ptr)
	    .setDataPointers(7, data_ptrs)
	    .setUids(7, uids)
	    .build();
	  DEBUG_CUDNN_MSG(log_buf, "variantPack " << variantPack.describe());
	  cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
	  checkCudnnErr(status);
	  cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
	}
    } catch (cudnn_frontend::cudnnException e) {
      std::cout << log_buf.str() << "[ERROR] Exception " << e.what() << std::endl;
    }
}

void
run_dconv_add_drelu_dscale(int64_t* x_dim_padded,
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
                       at::Half* devPtrR,
		       at::Half* devPtrI) {
    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;
    try {
        int convDim = 2;

        // Creates the necessary tensor descriptors
        dconv_add_descriptors tensors = create_dconv_add_descriptors(
            x_dim_padded, pad, convstride, dilation, w_dim_padded, y_dim_padded, dataType);
        DEBUG_CUDNN_MSG(log_buf, std::get<X_OR_DX_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DY_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<W_OR_DW_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<SCALE_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<RELU_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTER_DCONV_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTER_DRELU_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DGRAD_INPUT_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DGRAD_OPTIONAL_TENSOR>(tensors).describe());

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

        // optional add
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, addDesc.describe());

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

        // Create add Node.
        auto add_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(std::get<AFTER_DCONV_TENSOR>(tensors))
                           .setbDesc(std::get<DGRAD_INPUT_TENSOR>(tensors))
                           .setyDesc(std::get<DGRAD_OPTIONAL_TENSOR>(tensors))
                           .setpwDesc(addDesc)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, add_op.describe());

        // TODO: do we need getOutputTensor(), and what it returns in backward case?
        // Create an relu backward Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setdyDesc(std::get<DGRAD_OPTIONAL_TENSOR>(tensors))
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
        std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &add_op, &act_op, &scale_op};

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
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrZ, devPtrR, devPtrI};
        int64_t uids[]    = {'x', 'y', 'w', 's', 'r', 'i'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
          .setWorkspacePointer(workspace_ptr)
          .setDataPointers(6, data_ptrs)
          .setUids(6, uids)
          .build();
        DEBUG_CUDNN_MSG(log_buf, "variantPack " << variantPack.describe());
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        checkCudnnErr(status);
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
    } catch (cudnn_frontend::cudnnException e) {
      std::cout << log_buf.str() << "[ERROR] Exception " << e.what() << std::endl;
    }
}

void
run_dconv_drelu_dscale_mask(int64_t* x_dim_padded,
                            int64_t* pad,
			    int64_t* convstride,
			    int64_t* dilation,
			    int64_t* w_dim_padded,
			    int64_t* y_dim_padded,
			    int64_t* threshold_dim,
			    cudnnDataType_t dataType,
			    at::Half* devPtrX,
			    at::Half* devPtrW,
			    at::Half* devPtrY,
			    at::Half* devPtrZ,
			    at::Half* devPtrR,
			    int* devPtrT,
			    int* devPtrU,
			    int axis) {
    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;
    try {
        int convDim = 2;

        // Creates the necessary tensor descriptors
        dconv_mask_descriptors tensors = create_dconv_mask_descriptors(
            x_dim_padded, pad, convstride, dilation, w_dim_padded, y_dim_padded, threshold_dim, dataType);
        DEBUG_CUDNN_MSG(log_buf, std::get<X_OR_DX_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DY_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<W_OR_DW_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<SCALE_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<RELU_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTER_DCONV_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<AFTER_DRELU_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DGRAD_OPTIONAL_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DGRAD_GEN_INDEX_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DGRAD_MASK_TOP_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DGRAD_MASK_BOTTOM_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DGRAD_MASK_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DGRAD_THRESHOLD_TOP_TENSOR>(tensors).describe());
        DEBUG_CUDNN_MSG(log_buf, std::get<DGRAD_THRESHOLD_BOTTOM_TENSOR>(tensors).describe());

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

        // Define the genIndex descriptor
        auto genIndexDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_GEN_INDEX)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .setAxis(axis)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, genIndexDesc.describe());

        // Define the lessThan descriptor
        auto lessThanDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_CMP_LT)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, lessThanDesc.describe());

        // Define the greaterThan descriptor
        auto greaterThanDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_CMP_GT)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, greaterThanDesc.describe());

        // Define the logical_or descriptor
        auto logicalOrDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_LOGICAL_OR)
                           .setMathPrecision(CUDNN_DATA_BOOLEAN)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, logicalOrDesc.describe());

        // Define the binary_selection descriptor
        auto selectionDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_BINARY_SELECT)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, selectionDesc.describe());

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
          .setyDesc(std::get<DGRAD_OPTIONAL_TENSOR>(tensors))
          .setpwDesc(scaleDesc)
          .build();
        DEBUG_CUDNN_MSG(log_buf, scale_op.describe());

        // Create a Gen_Index Node.
        auto genIndex_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(std::get<DGRAD_OPTIONAL_TENSOR>(tensors))
                          .setyDesc(std::get<DGRAD_GEN_INDEX_TENSOR>(tensors))
                          .setpwDesc(genIndexDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, genIndex_op.describe());

        // Create a LessThan Node.
        auto lessThan_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(std::get<DGRAD_GEN_INDEX_TENSOR>(tensors))
                          .setbDesc(std::get<DGRAD_THRESHOLD_TOP_TENSOR>(tensors))
                          .setyDesc(std::get<DGRAD_MASK_TOP_TENSOR>(tensors))
                          .setpwDesc(lessThanDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, lessThan_op.describe());

        // Create a GreaterThan Node.
        auto greaterThan_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(std::get<DGRAD_GEN_INDEX_TENSOR>(tensors))
                          .setbDesc(std::get<DGRAD_THRESHOLD_BOTTOM_TENSOR>(tensors))
                          .setyDesc(std::get<DGRAD_MASK_BOTTOM_TENSOR>(tensors))
                          .setpwDesc(greaterThanDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, greaterThan_op.describe());

        // Create a LogicalOr Node.
        auto logicalOr_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(std::get<DGRAD_MASK_TOP_TENSOR>(tensors))
                          .setbDesc(std::get<DGRAD_MASK_BOTTOM_TENSOR>(tensors))
                          .setyDesc(std::get<DGRAD_MASK_TENSOR>(tensors))
                          .setpwDesc(logicalOrDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, logicalOr_op.describe());

        // Create a Binary_Selection Node.
        auto selection_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(std::get<AFTER_DCONV_TENSOR>(tensors))
                          .setbDesc(std::get<DGRAD_OPTIONAL_TENSOR>(tensors))
                          .settDesc(std::get<DGRAD_MASK_TENSOR>(tensors))
                          .setyDesc(std::get<X_OR_DX_TENSOR>(tensors))
                          .setpwDesc(selectionDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, selection_op.describe());

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 8> ops = {&conv_op, &act_op, &scale_op, &genIndex_op, &lessThan_op, &greaterThan_op, &logicalOr_op, &selection_op};

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
        void* data_ptrs[] = {devPtrX, devPtrY, devPtrW, devPtrZ, devPtrR, devPtrT, devPtrU};
        int64_t uids[]    = {'x', 'y', 'w', 's', 'r', 't', 'u'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
          .setWorkspacePointer(workspace_ptr)
          .setDataPointers(7, data_ptrs)
          .setUids(7, uids)
          .build();
        DEBUG_CUDNN_MSG(log_buf, "variantPack " << variantPack.describe());
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        checkCudnnErr(status);
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
    } catch (cudnn_frontend::cudnnException e) {
      std::cout << log_buf.str() << "[ERROR] Exception " << e.what() << std::endl;
    }
}

struct bottleneck_forward_status {

  int64_t dimA[4];
  int64_t filterdimA1[4];
  int64_t filterdimA2[4];
  int64_t filterdimA2hh[4];
  int64_t filterdimA3[4];
  int64_t filterdimA4[4];

  int64_t threshdim[4];

  int axis[4];

  int64_t outdimA0[4];
  int64_t outdimA1[4];
  int64_t outdimA1b[4]; // out1_pad
  int64_t outdimA2[4];
  int64_t outdimA3[4];
  int64_t outdimA4[4];

  int64_t padA[2];
  int64_t padA1[2];
  int64_t padA2[2];  // halo padding
  int64_t dilationA[2];
  int64_t convstrideA[2];
  int64_t convstride1X1[2];

  int64_t outdim0[4]; // halo input shape
  int64_t outdim1[4];
  int64_t outdim1b[4];
  int64_t outdim2[4];
  int64_t outdim3[4];
  int64_t outdim4[4]; // halo output shape

  void init(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs) {
    dimA[0] = dimA[1] = dimA[2] = dimA[3] = 0;
    filterdimA1[0] = filterdimA1[1] = filterdimA1[2] = filterdimA1[3] = 0;
    filterdimA2[0] = filterdimA2[1] = filterdimA2[2] = filterdimA2[3] = 0;
    filterdimA2hh[0] = filterdimA2hh[1] = filterdimA2hh[2] = filterdimA2hh[3] = 0;
    filterdimA3[0] = filterdimA3[1] = filterdimA3[2] = filterdimA3[3] = 0;
    filterdimA4[0] = filterdimA4[1] = filterdimA4[2] = filterdimA4[3] = 0;
    threshdim[0] = threshdim[1] = threshdim[2] = threshdim[3] = 1;

    // All dim calculation after this order of n,c,h,w
    if (explicit_nhwc) {
      axis[0] = 0;
      axis[1] = 3;
      axis[2] = 1;
      axis[3] = 2;
    } else {
      axis[0] = 0;
      axis[1] = 1;
      axis[2] = 2;
      axis[3] = 3;
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
    for (int dim=0;dim<4;dim++) {
      if (dim == 2) {
	filterdimA2hh[dim] = 1;
      } else {
	filterdimA2hh[dim] = filterdimA2[dim];
      }
    }

    // output dim in n,c,h,w used by backend
    outdimA0[0] = outdimA0[1] = outdimA0[2] = outdimA0[3] = 0;
    outdimA1[0] = outdimA1[1] = outdimA1[2] = outdimA1[3] = 0;
    outdimA1b[0] = outdimA1b[1] = outdimA1b[2] = outdimA1b[3] = 0;
    outdimA2[0] = outdimA2[1] = outdimA2[2] = outdimA2[3] = 0;
    outdimA3[0] = outdimA3[1] = outdimA3[2] = outdimA3[3] = 0;
    outdimA4[0] = outdimA4[1] = outdimA4[2] = outdimA4[3] = 0;

    // use these fixed value for test run
    padA[0] = 0; padA[1] = 0;
    padA1[0] = 1; padA1[1] = 1;
    padA2[0] = 0; padA2[1] = 1;
    dilationA[0] = 1; dilationA[1] = 1;
    convstrideA[0] = 1; convstrideA[1] = 1;
    convstride1X1[0] = stride_1X1; convstride1X1[1] = stride_1X1;

    // compute output from pad/stride/dilation
    outdimA1[0] = dimA[0];
    outdimA1[1] = filterdimA1[0];
    for (int dim = 0; dim < 2; dim++) {
      outdimA1[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA1[dim + 2], convstride1X1[dim], dilationA[dim]);
    }
    for (int dim = 0; dim < 4; dim++) {
      if (dim == 2) {
        outdimA1b[dim] = outdimA1[dim] + 2;
      } else {
        outdimA1b[dim] = outdimA1[dim];
      }
    }

    outdimA2[0] = outdimA1[0];
    outdimA2[1] = filterdimA2[0];
    for (int dim = 0; dim < 2; dim++) {
      outdimA2[dim + 2] = getFwdConvOutputDim(outdimA1[dim + 2], padA1[dim], filterdimA2[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    for (int dim = 0; dim < 4; dim++) {
      if (dim == 2) {
	outdimA0[dim] = 3;
	outdimA4[dim] = 1;
      } else {
        outdimA0[dim] = outdimA1[dim];
	outdimA4[dim] = outdimA2[dim];
      }
    }

    outdimA3[0] = outdimA2[0];
    outdimA3[1] = filterdimA3[0];
    for (int dim = 0; dim < 2; dim++) {
      outdimA3[dim + 2] = getFwdConvOutputDim(outdimA2[dim + 2], padA[dim], filterdimA3[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    // Create output tensor in the correct shape in pytorch's view
    outdim1[0] = outdim1[1] = outdim1[2] = outdim1[3] = 0;
    outdim1b[0] = outdim1b[1] = outdim1b[2] = outdim1b[3] = 0;
    outdim2[0] = outdim2[1] = outdim2[2] = outdim2[3] = 0;
    outdim3[0] = outdim3[1] = outdim3[2] = outdim3[3] = 0;
    if (explicit_nhwc) {
      axis[0] = 0;
      axis[1] = 2;
      axis[2] = 3;
      axis[3] = 1;
    }
    for (int dim=0;dim<4;dim++) {
      outdim0[dim] = outdimA0[axis[dim]];
      outdim1[dim] = outdimA1[axis[dim]];
      outdim1b[dim] = outdimA1b[axis[dim]];
      outdim2[dim] = outdimA2[axis[dim]];
      outdim3[dim] = outdimA3[axis[dim]];
      outdim4[dim] = outdimA4[axis[dim]];
    }
  }
};

bottleneck_forward_status forward_state;

} // end of anonymous namespace

std::vector<at::Tensor> bottleneck_forward_init(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs) {
  // NB! Bottleneck_forward and bottleneck_backward are NOT thread safe method.
  // NB! We use a global object to store state.
  forward_state.init(explicit_nhwc, stride_1X1, inputs);

  // create output vector
  std::vector<at::Tensor> outputs;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  //printf("outdim1 = (%d,%d,%d,%d)\n",forward_state.outdim1[0],forward_state.outdim1[1],forward_state.outdim1[2],forward_state.outdim1[3]);
  auto out1 = at::empty(forward_state.outdim1, inputs[0].type(), output_format);
  auto out2 = at::empty(forward_state.outdim2, inputs[0].type(), output_format);
  auto out3 = at::empty(forward_state.outdim3, inputs[0].type(), output_format);

  outputs.push_back(out1);
  outputs.push_back(out2);
  outputs.push_back(out3);

  return outputs;
}

// inputs contains x,w,z,b,(i)
void bottleneck_forward_out1(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs) {

  std::cout << std::fixed;

  // run
  at::Half* x = inputs[0].data_ptr<at::Half>();
  at::Half* w = inputs[1].data_ptr<at::Half>();
  at::Half* z = inputs[4].data_ptr<at::Half>();
  at::Half* b = inputs[7].data_ptr<at::Half>();
  auto out1 = outputs[0];
  at::Half* y1 = out1.data_ptr<at::Half>();

  run_conv_scale_bias_add_activation(forward_state.dimA,
                                     forward_state.padA,
                                     forward_state.convstride1X1,
                                     forward_state.dilationA,
                                     forward_state.filterdimA1,
                                     forward_state.outdimA1,
                                     CUDNN_DATA_HALF,
                                     x,
                                     w,
                                     y1,
                                     z,
                                     b,
                                     nullptr);

  DEBUG_MSG("[DEBUG] new relu1 : " << out1.to(at::kFloat).sum().item<float>());
}

// computes halo (top or bottom) from fat halo input.
// fat halo input is 3 pixels wide in H.
at::Tensor bottleneck_forward_out2_halo(bool explicit_nhwc, at::Tensor fat_halo_y1, std::vector<at::Tensor> inputs) {

  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // run
  at::Half* w = inputs[2].data_ptr<at::Half>();
  at::Half* z = inputs[5].data_ptr<at::Half>();
  at::Half* b = inputs[8].data_ptr<at::Half>();
  
  at::Half* y1 = fat_halo_y1.data_ptr<at::Half>();

  auto halo_y2 = at::empty(forward_state.outdim4, inputs[0].type(), output_format);
  at::Half* y2 = halo_y2.data_ptr<at::Half>();

  run_conv_scale_bias_add_activation(forward_state.outdimA0,
                                     forward_state.padA2,
                                     forward_state.convstrideA,
                                     forward_state.dilationA,
                                     forward_state.filterdimA2,
                                     forward_state.outdimA4,
                                     CUDNN_DATA_HALF,
                                     y1,
                                     w,
                                     y2,
                                     z,
                                     b,
                                     nullptr);

  return halo_y2;
}

// compute halo correction term (top or bottom) from slim halo input (N,C,1,W).
// slim halo input is 1 pixel wide in H.
at::Tensor bottleneck_forward_out2_halo_corr(bool explicit_nhwc, at::Tensor slim_halo_y1, std::vector<at::Tensor> inputs, at::Tensor w1by3, at::Tensor out2_part_halo) {

  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // run
  at::Half* w = w1by3.data_ptr<at::Half>();  // C,C,1,3
  at::Half* z = inputs[5].data_ptr<at::Half>();
  at::Half* b = inputs[8].data_ptr<at::Half>();

  at::Half* y1 = slim_halo_y1.data_ptr<at::Half>();

  at::Half* prev_out2 = out2_part_halo.data_ptr<at::Half>();

  auto halo_y2 = at::empty(forward_state.outdim4, inputs[0].type(), output_format);
  at::Half* y2 = halo_y2.data_ptr<at::Half>();

  run_conv_add_scale_bias_activation(forward_state.outdimA4,
                                     forward_state.padA2,
                                     forward_state.convstrideA,
                                     forward_state.dilationA,
                                     forward_state.filterdimA2hh,
                                     forward_state.outdimA4,
                                     CUDNN_DATA_HALF,
                                     y1,
                                     w,
                                     y2,
                                     z,
                                     b,
				     prev_out2);

  return halo_y2;
}

void bottleneck_forward_out2(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs) {

  std::cout << std::fixed;

  // from _out1 method
  at::Half* x = inputs[0].data_ptr<at::Half>();
  auto out1 = outputs[0];
  at::Half* y1 = out1.data_ptr<at::Half>();

  // run
  at::Half* w = inputs[2].data_ptr<at::Half>();
  at::Half* z = inputs[5].data_ptr<at::Half>();
  at::Half* b = inputs[8].data_ptr<at::Half>();
  auto out2 = outputs[1];
  at::Half* y2 = out2.data_ptr<at::Half>();

  //printf("forward_state.outdimA1 = {%d,%d,%d,%d}\n",forward_state.outdimA1[0],forward_state.outdimA1[1],forward_state.outdimA1[2],forward_state.outdimA1[3]);
  //printf("forward_state.padA1 = {%d,%d}\n",forward_state.padA1[0],forward_state.padA1[1]);
  //printf("forward_state.convstrideA = {%d,%d}\n",forward_state.convstrideA[0],forward_state.convstrideA[1]);
  //printf("forward_state.dilationA = {%d,%d}\n",forward_state.dilationA[0],forward_state.dilationA[1]);
  //printf("forward_state.filterdimA2 = {%d,%d,%d,%d}\n",forward_state.filterdimA2[0],forward_state.filterdimA2[1],forward_state.filterdimA2[2],forward_state.filterdimA2[3]);
  //printf("forward_state.outdimA2 = {%d,%d,%d,%d}\n",forward_state.outdimA2[0],forward_state.outdimA2[1],forward_state.outdimA2[2],forward_state.outdimA2[3]);
  run_conv_scale_bias_add_activation(forward_state.outdimA1,
                                     forward_state.padA1,
                                     forward_state.convstrideA,
                                     forward_state.dilationA,
                                     forward_state.filterdimA2,
                                     forward_state.outdimA2,
                                     CUDNN_DATA_HALF,
                                     y1,
                                     w,
                                     y2,
                                     z,
                                     b,
                                     nullptr);
  DEBUG_MSG("[DEBUG] new relu2 : " << out2.to(at::kFloat).sum().item<float>());
}

void bottleneck_forward_out2_mask(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs, at::Tensor thresholdTop, at::Tensor thresholdBottom) {

  std::cout << std::fixed;

  // from _out1 method
  at::Half* x = inputs[0].data_ptr<at::Half>();
  auto out1 = outputs[0];
  at::Half* y1 = out1.data_ptr<at::Half>();

  // run
  at::Half* w = inputs[2].data_ptr<at::Half>();
  at::Half* z = inputs[5].data_ptr<at::Half>();
  at::Half* b = inputs[8].data_ptr<at::Half>();
  auto out2 = outputs[1];
  at::Half* y2 = out2.data_ptr<at::Half>();

  //printf("forward_state.outdimA1 = {%d,%d,%d,%d}\n",forward_state.outdimA1[0],forward_state.outdimA1[1],forward_state.outdimA1[2],forward_state.outdimA1[3]);
  //printf("forward_state.padA1 = {%d,%d}\n",forward_state.padA1[0],forward_state.padA1[1]);
  //printf("forward_state.convstrideA = {%d,%d}\n",forward_state.convstrideA[0],forward_state.convstrideA[1]);
  //printf("forward_state.dilationA = {%d,%d}\n",forward_state.dilationA[0],forward_state.dilationA[1]);
  //printf("forward_state.filterdimA2 = {%d,%d,%d,%d}\n",forward_state.filterdimA2[0],forward_state.filterdimA2[1],forward_state.filterdimA2[2],forward_state.filterdimA2[3]);
  //printf("forward_state.outdimA2 = {%d,%d,%d,%d}\n",forward_state.outdimA2[0],forward_state.outdimA2[1],forward_state.outdimA2[2],forward_state.outdimA2[3]);
  run_conv_scale_bias_add_activation_mask(forward_state.outdimA1,
                                          forward_state.padA1,
					  forward_state.convstrideA,
					  forward_state.dilationA,
					  forward_state.filterdimA2,
					  forward_state.outdimA2,
					  forward_state.threshdim,
					  CUDNN_DATA_HALF,
					  y1,
					  w,
					  y2,
					  z,
					  b,
					  nullptr,
					  thresholdTop.data_ptr<int>(),
                                          thresholdBottom.data_ptr<int>(),
					  2);  // axis == 1 -> Does this assume explicit NHWC?
  DEBUG_MSG("[DEBUG] new relu2 : " << out2.to(at::kFloat).sum().item<float>());
}

void bottleneck_forward_out2_pad(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs, at::Tensor out1_pad) {

  std::cout << std::fixed;

  // from _out1 method
  at::Half* x = inputs[0].data_ptr<at::Half>();
  auto out1 = outputs[0];
  at::Half* y1 = out1_pad.data_ptr<at::Half>();

  // run
  at::Half* w = inputs[2].data_ptr<at::Half>();
  at::Half* z = inputs[5].data_ptr<at::Half>();
  at::Half* b = inputs[8].data_ptr<at::Half>();
  auto out2 = outputs[1];
  at::Half* y2 = out2.data_ptr<at::Half>();

  //printf("forward_state.outdimA1 = {%d,%d,%d,%d}\n",forward_state.outdimA1[0],forward_state.outdimA1[1],forward_state.outdimA1[2],forward_state.outdimA1[3]);
  //printf("forward_state.padA1 = {%d,%d}\n",forward_state.padA1[0],forward_state.padA1[1]);
  //printf("forward_state.convstrideA = {%d,%d}\n",forward_state.convstrideA[0],forward_state.convstrideA[1]);
  //printf("forward_state.dilationA = {%d,%d}\n",forward_state.dilationA[0],forward_state.dilationA[1]);
  //printf("forward_state.filterdimA2 = {%d,%d,%d,%d}\n",forward_state.filterdimA2[0],forward_state.filterdimA2[1],forward_state.filterdimA2[2],forward_state.filterdimA2[3]);
  //printf("forward_state.outdimA2 = {%d,%d,%d,%d}\n",forward_state.outdimA2[0],forward_state.outdimA2[1],forward_state.outdimA2[2],forward_state.outdimA2[3]);
  run_conv_scale_bias_add_activation(forward_state.outdimA1b,
                                     forward_state.padA2,
                                     forward_state.convstrideA,
                                     forward_state.dilationA,
                                     forward_state.filterdimA2,
                                     forward_state.outdimA2,
                                     CUDNN_DATA_HALF,
                                     y1,
                                     w,
                                     y2,
                                     z,
                                     b,
                                     nullptr);
  DEBUG_MSG("[DEBUG] new relu2 : " << out2.to(at::kFloat).sum().item<float>());
}

void bottleneck_forward_rest(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs) {

  std::cout << std::fixed;

  // from _out1 method
  at::Half* x = inputs[0].data_ptr<at::Half>();

  // create output of conv3
  auto out3 = outputs[2];
  at::Half* y3 = out3.data_ptr<at::Half>();

  // create output of conv4 that may exist
  auto identity = at::empty_like(out3);
  at::Half* yi = identity.data_ptr<at::Half>();

  at::Half *w, *z, *b;

  if (stride_1X1 != 1 || forward_state.filterdimA3[0] != forward_state.dimA[1]){

    w = inputs[10].data_ptr<at::Half>();
    z = inputs[11].data_ptr<at::Half>();
    b = inputs[12].data_ptr<at::Half>();
    run_conv_scale_bias(forward_state.dimA,
                        forward_state.padA,
                        forward_state.convstride1X1,
                        forward_state.dilationA,
                        forward_state.filterdimA4,
                        forward_state.outdimA3,
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

  auto out2 = outputs[1];
  at::Half* y2 = out2.data_ptr<at::Half>();

  w = inputs[3].data_ptr<at::Half>();
  z = inputs[6].data_ptr<at::Half>();
  b = inputs[9].data_ptr<at::Half>();

  run_conv_scale_bias_add_activation(forward_state.outdimA2,
                                     forward_state.padA,
                                     forward_state.convstrideA,
                                     forward_state.dilationA,
                                     forward_state.filterdimA3,
                                     forward_state.outdimA3,
                                     CUDNN_DATA_HALF,
                                     y2,
                                     w,
                                     y3,
                                     z,
                                     b,
                                     yi);
  DEBUG_MSG("[DEBUG] new relu3 : " << out3.to(at::kFloat).sum().item<float>());
}

namespace {

struct bottleneck_backward_state {

  int64_t dimA[4];
  int64_t filterdimA1[4];
  int64_t filterdimA2[4];
  int64_t filterdimA3[4];
  int64_t filterdimA4[4];
  int64_t filterdimA2hh[4]; // Cin,Cout,1,3
  int64_t threshdim[4];

  int axis[4];

  int64_t outdimA1[4]; // grad_out1
  int64_t outdimA1b[4]; // out1_pad
  int64_t outdimA2[4]; // grad_out2
  int64_t outdimA3[4];
  int64_t outdimA1h[4]; // output: grad_out1 halo (H=3)
  int64_t outdimA2h[4]; // input : grad_out2 halo cells (H=3)
  int64_t outdimA1hh[4]; // input: grad_out2 halo (H=1)
  int64_t outdimA2hh[4]; // input: out1 halo (H=1)

  int64_t padA[2];
  int64_t padA1[2];
  int64_t padA2[2];
  int64_t dilationA[2];
  int64_t convstrideA[2];
  int64_t convstride1X1[2];

  int64_t filterdim2hh[4]; // Cin,1,3,Cout

  int64_t outdim1[4];
  int64_t outdim1b[4];
  int64_t outdim2[4];
  int64_t outdim3[4];
  int64_t outdim1h[4];
  int64_t outdim1hh[4];

  void init(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs) {
    // setup dimensions
    dimA[0] = dimA[1] = dimA[2] = dimA[3] = 0;
    filterdimA1[0] = filterdimA1[1] = filterdimA1[2] = filterdimA1[3] = 0;
    filterdimA2[0] = filterdimA2[1] = filterdimA2[2] = filterdimA2[3] = 0;
    filterdimA3[0] = filterdimA3[1] = filterdimA3[2] = filterdimA3[3] = 0;
    filterdimA4[0] = filterdimA4[1] = filterdimA4[2] = filterdimA4[3] = 0;
    filterdimA2hh[0] = filterdimA2hh[1] = filterdimA2hh[2] = filterdimA2hh[3] = 0;
    threshdim[0] = threshdim[1] = threshdim[2] = threshdim[3] = 1;

    // All dim calculation after this order of n,c,h,w
    if (explicit_nhwc) {
      axis[0] = 0;
      axis[1] = 3;
      axis[2] = 1;
      axis[3] = 2;
    } else {
      axis[0] = 0;
      axis[1] = 1;
      axis[2] = 2;
      axis[3] = 3;
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

    for (int dim=0;dim<4;dim++) {
      if (dim == 2) {
        filterdimA2hh[dim] = 1;
      } else {
        filterdimA2hh[dim] = filterdimA2[dim];
      }
    }

    // output dim in n,c,h,w used by backend
    outdimA1[0] = outdimA1[1] = outdimA1[2] = outdimA1[3] = 0;
    outdimA1b[0] = outdimA1b[1] = outdimA1b[2] = outdimA1b[3] = 0;
    outdimA2[0] = outdimA2[1] = outdimA2[2] = outdimA2[3] = 0;
    outdimA3[0] = outdimA3[1] = outdimA3[2] = outdimA3[3] = 0;
    outdimA1h[0] = outdimA1h[1] = outdimA1h[2] = outdimA1h[3] = 0;
    outdimA2h[0] = outdimA2h[1] = outdimA2h[2] = outdimA2h[3] = 0;
    outdimA1hh[0] = outdimA1hh[1] = outdimA1hh[2] = outdimA1hh[3] = 0;
    outdimA2hh[0] = outdimA2hh[1] = outdimA2hh[2] = outdimA2hh[3] = 0;

    // use these fixed value for test run
    padA[0] = 0; padA[1] = 0;
    padA1[0] = 1; padA1[1] = 1;
    padA2[0] = 0; padA2[1] = 1;
    dilationA[0] = 1; dilationA[1] = 1;
    convstrideA[0] = 1; convstrideA[1] = 1;
    convstride1X1[0] = stride_1X1; convstride1X1[1] = stride_1X1;

    // compute output from pad/stride/dilation
    outdimA1[0] = dimA[0];
    outdimA1[1] = filterdimA1[0];
    for (int dim = 0; dim < 2; dim++) {
      outdimA1[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA1[dim + 2], convstride1X1[dim], dilationA[dim]);
    }
    for (int dim = 0; dim < 4; dim++) {
      if (dim == 2) {
	outdimA1b[dim] = outdimA1[dim] + 2;
      } else {
	outdimA1b[dim] = outdimA1[dim];
      }
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

    for (int dim = 0; dim < 4; dim++) {
      if (dim == 2) {
	outdimA1h[dim] = 3;
	outdimA2h[dim] = 3;
        outdimA1hh[dim] = 1;
        outdimA2hh[dim] = 1;
      } else {
	outdimA1h[dim] = outdimA1[dim];
	outdimA2h[dim] = outdimA2[dim];
        outdimA1hh[dim] = outdimA1[dim];
        outdimA2hh[dim] = outdimA2[dim];
      }
    }

    // Create output tensor in the correct shape in pytorch's view
    outdim1[0] = outdim1[1] = outdim1[2] = outdim1[3] = 0;
    outdim1b[0] = outdim1b[1] = outdim1b[2] = outdim1b[3] = 0;
    outdim2[0] = outdim2[1] = outdim2[2] = outdim2[3] = 0;
    outdim3[0] = outdim3[1] = outdim3[2] = outdim3[3] = 0;
    outdim1h[0] = outdim1h[1] = outdim1h[2] = outdim1h[3] = 0;
    outdim1hh[0] = outdim1hh[1] = outdim1hh[2] = outdim1hh[3] = 0;
    filterdim2hh[0] = filterdim2hh[1] = filterdim2hh[2] = filterdim2hh[3] = 0;
    if (explicit_nhwc) {
      axis[0] = 0;
      axis[1] = 2;
      axis[2] = 3;
      axis[3] = 1;
    }
    for (int dim=0;dim<4;dim++) {
      outdim1[dim] = outdimA1[axis[dim]];
      outdim1b[dim] = outdimA1b[axis[dim]];
      outdim2[dim] = outdimA2[axis[dim]];
      outdim3[dim] = outdimA3[axis[dim]];
      outdim1h[dim] = outdimA1h[axis[dim]];
      outdim1hh[dim] = outdimA1hh[axis[dim]];
      filterdim2hh[dim] = filterdimA2hh[axis[dim]];
    }
  }
};

bottleneck_backward_state backward_state;

}

std::vector<at::Tensor> bottleneck_backward_init(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs) {

  std::cout << std::fixed;

  backward_state.init(explicit_nhwc, stride_1X1, inputs);

  // create output vector
  std::vector<at::Tensor> outputs;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  auto grad_x = at::empty_like(inputs[0]);
  auto wgrad1 = at::empty_like(inputs[1]);
  auto wgrad2 = at::empty_like(inputs[2]);
  auto wgrad3 = at::empty_like(inputs[3]);

  outputs.push_back(grad_x);
  outputs.push_back(wgrad1);
  outputs.push_back(wgrad2);
  outputs.push_back(wgrad3);
  if (stride_1X1 != 1 || backward_state.filterdimA3[0] != backward_state.dimA[1]) {
    auto wgrad4 = at::empty_like(inputs[14]);
    outputs.push_back(wgrad4);
  }

  return outputs;
}

void bottleneck_backward_wgrad3(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs) {

  // dconv3+drelu2+dscale2
  at::Half* conv_in = inputs[13].data_ptr<at::Half>();
  at::Half* dy3 = inputs[10].data_ptr<at::Half>();

  // wgrad
  auto wgrad3 = outputs[3];
  at::Half* dw3 = wgrad3.data_ptr<at::Half>();
  run_dconv(backward_state.outdimA2,
            backward_state.padA,
            backward_state.convstrideA,
            backward_state.dilationA,
            backward_state.filterdimA3,
            backward_state.outdimA3,
            CUDNN_DATA_HALF,
            conv_in,
            dw3,
            dy3,
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
  DEBUG_MSG("[DEBUG] new wgrad3 : " << wgrad3.to(at::kFloat).sum().item<float>());

}

at::Tensor bottleneck_backward_grad_out2(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs) {

  bool requires_grad = inputs[0].requires_grad();

  std::cout << std::fixed;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // dconv3+drelu2+dscale2
  at::Half* conv_in = inputs[13].data_ptr<at::Half>();
  at::Half* dy3 = inputs[10].data_ptr<at::Half>();

  DEBUG_MSG("[DEBUG] new dconv3 : " << inputs[10].to(at::kFloat).sum().item<float>());

  // dgrad
  auto grad_out2 = at::empty(backward_state.outdim2, inputs[0].type(), output_format);
  at::Half* dy2 = grad_out2.data_ptr<at::Half>();
  at::Half* w = inputs[3].data_ptr<at::Half>();
  at::Half* z = inputs[5].data_ptr<at::Half>();

  at::Half* relu2 = inputs[13].data_ptr<at::Half>();

  run_dconv_drelu_dscale(backward_state.outdimA2,
                         backward_state.padA,
                         backward_state.convstrideA,
                         backward_state.dilationA,
                         backward_state.filterdimA3,
                         backward_state.outdimA3,
                         CUDNN_DATA_HALF,
                         dy2,
                         w,
                         dy3,
                         z,
                         relu2);

  // do halo exchange of dy2 here

  DEBUG_MSG("[DEBUG] new dconv2 : " << grad_out2.to(at::kFloat).sum().item<float>());

  return grad_out2;
}

at::Tensor bottleneck_backward_grad_out1(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs, at::Tensor grad_out2) {

  bool requires_grad = inputs[0].requires_grad();

  std::cout << std::fixed;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // dgrad
  at::Half* dy2 = grad_out2.data_ptr<at::Half>();

  // dgrad
  auto grad_out1 = at::empty(backward_state.outdim1, inputs[0].type(), output_format);
  at::Half* dy1 = grad_out1.data_ptr<at::Half>();
  at::Half* w = inputs[2].data_ptr<at::Half>();
  at::Half* z = inputs[4].data_ptr<at::Half>();

  at::Half* relu1 = inputs[12].data_ptr<at::Half>();
  //printf("relu.shape = [%d,%d,%d,%d]\n",inputs[12].size(0),inputs[12].size(1),inputs[12].size(2),inputs[12].size(3));

  // fused dgrad
  //printf("backward_state.outdim1 = {%d,%d,%d,%d}\n",backward_state.outdim1[0],backward_state.outdim1[1],backward_state.outdim1[2],backward_state.outdim1[3]);
  run_dconv_drelu_dscale(backward_state.outdimA1,
                         backward_state.padA1,
                         backward_state.convstrideA,
                         backward_state.dilationA,
                         backward_state.filterdimA2,
                         backward_state.outdimA2,
                         CUDNN_DATA_HALF,
                         dy1,
                         w,
                         dy2,
                         z,
                         relu1);

  return grad_out1;
}

at::Tensor bottleneck_backward_grad_out1_mask(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs, at::Tensor grad_out2, at::Tensor thresholdTop, at::Tensor thresholdBottom) {

  bool requires_grad = inputs[0].requires_grad();

  std::cout << std::fixed;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // dgrad
  at::Half* dy2 = grad_out2.data_ptr<at::Half>();

  // dgrad
  auto grad_out1 = at::empty(backward_state.outdim1, inputs[0].type(), output_format);
  at::Half* dy1 = grad_out1.data_ptr<at::Half>();
  at::Half* w = inputs[2].data_ptr<at::Half>();
  at::Half* z = inputs[4].data_ptr<at::Half>();

  at::Half* relu1 = inputs[12].data_ptr<at::Half>();
  //printf("relu.shape = [%d,%d,%d,%d]\n",inputs[12].size(0),inputs[12].size(1),inputs[12].size(2),inputs[12].size(3));

  // fused dgrad
  run_dconv_drelu_dscale_mask(backward_state.outdimA1,
                              backward_state.padA1,
                              backward_state.convstrideA,
                              backward_state.dilationA,
                              backward_state.filterdimA2,
                              backward_state.outdimA2,
			      backward_state.threshdim,
                              CUDNN_DATA_HALF,
                              dy1,
                              w,
                              dy2,
                              z,
                              relu1,
		              thresholdTop.data_ptr<int>(),
                              thresholdBottom.data_ptr<int>(),
			      2);

  return grad_out1;
}

// perform backward data 1x3 convolution (grad_out * w_rot180) on grad_out2 input of shape [N,1,W,C] with padding=(0,1) to produce output of shape [N,1,W,C]
at::Tensor bottleneck_backward_grad_out1_halo_corr(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, at::Tensor w1by3, std::vector<at::Tensor> outputs, at::Tensor grad_out2_halo, at::Tensor relu1_halo, at::Tensor part_grad_out1) {

  bool requires_grad = inputs[0].requires_grad();

  std::cout << std::fixed;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // dgrad
  at::Half* dy2h = grad_out2_halo.data_ptr<at::Half>();

  // dgrad
  auto grad_out1_halo = at::empty(backward_state.outdim1hh, inputs[0].type(), output_format);
  at::Half* dy1h = grad_out1_halo.data_ptr<at::Half>();
  //at::Half* w = inputs[2].data_ptr<at::Half>();  // use w1by3 instead, which is a sliced version of inputs[2]
  at::Half* w = w1by3.data_ptr<at::Half>();
  at::Half* z = inputs[4].data_ptr<at::Half>();
  at::Half* relu1h = relu1_halo.data_ptr<at::Half>();
  at::Half* pdy1h = part_grad_out1.data_ptr<at::Half>();

  //printf("relu.shape = [%d,%d,%d,%d]\n",relu1_halo.size(0),relu1_halo.size(1),relu1_halo.size(2),relu1_halo.size(3));
  // fused dgrad
  //printf("backward_state.outdimA1h = {%d,%d,%d,%d}\n",backward_state.outdimA1h[0],backward_state.outdimA1h[1],backward_state.outdimA1h[2],backward_state.outdimA1h[3]);
  //printf("backward_state.outdimA2h = {%d,%d,%d,%d}\n",backward_state.outdimA2h[0],backward_state.outdimA2h[1],backward_state.outdimA2h[2],backward_state.outdimA2h[3]);
  //printf("backward_state.filterdimA2 = {%d,%d,%d,%d}\n",backward_state.filterdimA2[0],backward_state.filterdimA2[1],backward_state.filterdimA2[2],backward_state.filterdimA2[3]);
  run_dconv_add_drelu_dscale(backward_state.outdimA1hh,
                             backward_state.padA2, // 0,1
			     backward_state.convstrideA,
			     backward_state.dilationA,
			     backward_state.filterdimA2hh, // C,1,3,C
			     backward_state.outdimA2hh,
			     CUDNN_DATA_HALF,
			     dy1h,
			     w,
			     dy2h,
			     z,
			     relu1h,
			     pdy1h);

  return grad_out1_halo;
}

// perform backward data 3x3 convolution (grad_out * w_rot180) on grad_out2 input of shape [N,3,W,C] with padding=(1,1) to produce output of shape [N,3,W,C]
at::Tensor bottleneck_backward_grad_out1_halo(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs, at::Tensor grad_out2_halo, at::Tensor relu1_halo) {

  bool requires_grad = inputs[0].requires_grad();

  std::cout << std::fixed;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // dgrad
  at::Half* dy2h = grad_out2_halo.data_ptr<at::Half>();

  // dgrad
  auto grad_out1_halo = at::empty(backward_state.outdim1h, inputs[0].type(), output_format);
  at::Half* dy1h = grad_out1_halo.data_ptr<at::Half>();
  at::Half* w = inputs[2].data_ptr<at::Half>();
  at::Half* z = inputs[4].data_ptr<at::Half>();

  at::Half* relu1h = relu1_halo.data_ptr<at::Half>();
  //printf("relu.shape = [%d,%d,%d,%d]\n",relu1_halo.size(0),relu1_halo.size(1),relu1_halo.size(2),relu1_halo.size(3));
  // fused dgrad
  //printf("backward_state.outdimA1h = {%d,%d,%d,%d}\n",backward_state.outdimA1h[0],backward_state.outdimA1h[1],backward_state.outdimA1h[2],backward_state.outdimA1h[3]);
  //printf("backward_state.outdimA2h = {%d,%d,%d,%d}\n",backward_state.outdimA2h[0],backward_state.outdimA2h[1],backward_state.outdimA2h[2],backward_state.outdimA2h[3]);
  //printf("backward_state.filterdimA2 = {%d,%d,%d,%d}\n",backward_state.filterdimA2[0],backward_state.filterdimA2[1],backward_state.filterdimA2[2],backward_state.filterdimA2[3]);
  run_dconv_drelu_dscale(backward_state.outdimA1h,
                         backward_state.padA1,
                         backward_state.convstrideA,
                         backward_state.dilationA,
                         backward_state.filterdimA2,
                         backward_state.outdimA2h,
                         CUDNN_DATA_HALF,
                         dy1h,
                         w,
                         dy2h,
                         z,
                         relu1h);

  return grad_out1_halo;
}

void bottleneck_backward_wgrad2_pad(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs, at::Tensor input, at::Tensor grad_out2) {

  std::cout << std::fixed;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // dgrad
  at::Half* dy2 = grad_out2.data_ptr<at::Half>();

  // dconv2+drelu1+dscale1
  at::Half* conv_in = input.data_ptr<at::Half>();

  // wgrad
  auto wgrad2 = outputs[2];
  at::Half* dw2 = wgrad2.data_ptr<at::Half>();

  //printf("outdimA1b = (%d,%d,%d,%d)\n",backward_state.outdimA1b[0],backward_state.outdimA1b[1],backward_state.outdimA1b[2],backward_state.outdimA1b[3]);
  //printf("backward_state.padA2 = {%d,%d}\n",backward_state.padA2[0],backward_state.padA2[1]);
  run_dconv(backward_state.outdimA1b,	// conv_in.shape (including H halos)
            backward_state.padA2,	// 0, 1
            backward_state.convstrideA,
            backward_state.dilationA,
            backward_state.filterdimA2, // dw2.shape
            backward_state.outdimA2,	// dy2.shape
            CUDNN_DATA_HALF,
            conv_in,
            dw2,
            dy2,
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
  DEBUG_MSG("[DEBUG] new wgrad2 : " << wgrad2.to(at::kFloat).sum().item<float>());
}

void bottleneck_backward_wgrad2(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs, at::Tensor grad_out2) {

  bool requires_grad = inputs[0].requires_grad();

  std::cout << std::fixed;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // dgrad
  at::Half* dy2 = grad_out2.data_ptr<at::Half>();

  // dconv2+drelu1+dscale1
  at::Half* conv_in = inputs[12].data_ptr<at::Half>();

  // wgrad
  auto wgrad2 = outputs[2];
  at::Half* dw2 = wgrad2.data_ptr<at::Half>();
  
  //printf("outdimA1 = (%d,%d,%d,%d)\n",backward_state.outdimA1[0],backward_state.outdimA1[1],backward_state.outdimA1[2],backward_state.outdimA1[3]);
  run_dconv(backward_state.outdimA1,
            backward_state.padA1,
            backward_state.convstrideA,
            backward_state.dilationA,
            backward_state.filterdimA2,
            backward_state.outdimA2,
            CUDNN_DATA_HALF,
            conv_in,
            dw2,
            dy2,
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
  DEBUG_MSG("[DEBUG] new wgrad2 : " << wgrad2.to(at::kFloat).sum().item<float>());
}

// compute halo cells for input volume of dimension [N,1,W,C] with padding=(0,1) to produce output volume of dimension [N,1,W,C]
// input and grad_out2_halo tensors are all of same shape
// output tensor is of shape [Cin,1,3,Cout] (regular filter dims are [Cin,3,3,Cout]
at::Tensor bottleneck_backward_wgrad2_halo(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs, at::Tensor input, at::Tensor grad_out2_halo) {

  bool requires_grad = inputs[0].requires_grad();

  std::cout << std::fixed;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // dgrad
  at::Half* dy2 = grad_out2_halo.data_ptr<at::Half>();

  // dconv2+drelu1+dscale1
  at::Half* conv_in = input.data_ptr<at::Half>();

  // wgrad
  auto wgrad2_halo = at::empty(backward_state.filterdim2hh, input.type(), output_format);
  at::Half* dw2 = wgrad2_halo.data_ptr<at::Half>();

  //printf("backward_state.outdimA1hh = {%d,%d,%d,%d}\n",backward_state.outdimA1hh[0],backward_state.outdimA1hh[1],backward_state.outdimA1hh[2],backward_state.outdimA1hh[3]);
  //printf("backward_state.outdimA2hh = {%d,%d,%d,%d}\n",backward_state.outdimA2hh[0],backward_state.outdimA2hh[1],backward_state.outdimA2hh[2],backward_state.outdimA2hh[3]);
  //printf("backward_state.filterdim2hh = {%d,%d,%d,%d}\n",backward_state.filterdim2hh[0],backward_state.filterdim2hh[1],backward_state.filterdim2hh[2],backward_state.filterdim2hh[3]);
  //printf("backward_state.filterdimA2hh = {%d,%d,%d,%d}\n",backward_state.filterdimA2hh[0],backward_state.filterdimA2hh[1],backward_state.filterdimA2hh[2],backward_state.filterdimA2hh[3]);
  //printf("backward_state.padA2 = {%d,%d}\n",backward_state.padA2[0],backward_state.padA2[1]);
  run_dconv(backward_state.outdimA1hh,  // N,C,1,W
            backward_state.padA2, // 0, 1
            backward_state.convstrideA,
            backward_state.dilationA,
            backward_state.filterdimA2hh,  // Cin,Cout,1,3
            backward_state.outdimA2hh,  // N,C,1,W
            CUDNN_DATA_HALF,
            conv_in,
            dw2,
            dy2,
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);

  return wgrad2_halo;
}

void bottleneck_backward_wgrad1(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs, at::Tensor grad_out1) {

  at::Half* x = inputs[0].data_ptr<at::Half>();
  at::Half* dy1 = grad_out1.data_ptr<at::Half>();

  // dconv1+add
  // wgrad
  auto wgrad1 = outputs[1];
  at::Half* dw1 = wgrad1.data_ptr<at::Half>();
  run_dconv(backward_state.dimA,
            backward_state.padA,
            backward_state.convstride1X1,
            backward_state.dilationA,
            backward_state.filterdimA1,
            backward_state.outdimA1,
            CUDNN_DATA_HALF,
            x,
            dw1,
            dy1,
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);

}

void bottleneck_backward_rest(bool explicit_nhwc, int stride_1X1, std::vector<at::Tensor> inputs, std::vector<at::Tensor> outputs, at::Tensor grad_out2, at::Tensor grad_out1) {

  bool requires_grad = inputs[0].requires_grad();

  std::cout << std::fixed;
  auto output_format = explicit_nhwc ? at::MemoryFormat::Contiguous : at::MemoryFormat::ChannelsLast;

  // dgrad
  at::Half* dy2 = grad_out2.data_ptr<at::Half>();
  at::Half* dy1 = grad_out1.data_ptr<at::Half>();

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

  at::Half* w = NULL;

  if (stride_1X1 != 1 || backward_state.filterdimA3[0] != backward_state.dimA[1]){
    w = inputs[14].data_ptr<at::Half>();
    at::Half* dy_conv4 = inputs[11].data_ptr<at::Half>();
    if (requires_grad) {
      run_dconv(backward_state.dimA,
                backward_state.padA,
                backward_state.convstride1X1,
                backward_state.dilationA,
                backward_state.filterdimA4,
                backward_state.outdimA3,
                CUDNN_DATA_HALF,
                dx_conv4,
                w,
                dy_conv4,
                CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
      // we don't print here since we can't hook out this grad in pytorch alone to compare, due to addition with dx
      // DEBUG_MSG("[DEBUG] new dx_identity : " << grad_x_conv4.to(at::kFloat).sum().item<float>());
    }
    // wgrad
    wgrad4 = outputs[4];
    at::Half* dw4 = wgrad4.data_ptr<at::Half>();
    run_dconv(backward_state.dimA,
              backward_state.padA,
              backward_state.convstride1X1,
              backward_state.dilationA,
              backward_state.filterdimA4,
              backward_state.outdimA3,
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

  // dgrad
  w = inputs[1].data_ptr<at::Half>();
  auto grad_x = outputs[0];
  at::Half* dx = grad_x.data_ptr<at::Half>();

  // backward strided conv cannot be fused
  // if stride == 1 but channel changes, we can fuse here
  if (requires_grad){
    if (stride_1X1 != 1){
      run_dconv(backward_state.dimA,
                backward_state.padA,
                backward_state.convstride1X1,
                backward_state.dilationA,
                backward_state.filterdimA1,
                backward_state.outdimA1,
                CUDNN_DATA_HALF,
                dx,
                w,
                dy1,
                CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
      // add 2 together
      grad_x.add_(grad_x_conv4);
    }
    else {
      run_dconv_add(backward_state.dimA,
                    backward_state.padA,
                    backward_state.convstride1X1,
                    backward_state.dilationA,
                    backward_state.filterdimA1,
                    backward_state.outdimA1,
                    CUDNN_DATA_HALF,
                    dx,
                    w,
                    dy1,
                    dx_conv4);
    }
  }

  DEBUG_MSG("[DEBUG] new dx : " << grad_x.to(at::kFloat).sum().item<float>());
  DEBUG_MSG("[DEBUG] new wgrad1 : " << wgrad1.to(at::kFloat).sum().item<float>());

  if (stride_1X1 != 1 || backward_state.filterdimA3[0] != backward_state.dimA[1]) {
    DEBUG_MSG("[DEBUG] new wgrad4 : " << wgrad4.to(at::kFloat).sum().item<float>());
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bottleneck_forward, "Bottleneck block forward");
  m.def("backward", &bottleneck_backward, "Bottleneck block backward");
  m.def("forward_init", &bottleneck_forward_init, "Bottleneck block init");
  m.def("forward_out1", &bottleneck_forward_out1, "Bottleneck block forward");
  m.def("forward_out2", &bottleneck_forward_out2, "Bottleneck block forward");
  m.def("forward_out2_mask", &bottleneck_forward_out2_mask, "Bottleneck block forward");
  m.def("forward_out2_halo", &bottleneck_forward_out2_halo, "Bottleneck block forward");
  m.def("forward_out2_halo_corr", &bottleneck_forward_out2_halo_corr, "Bottleneck block forward");
  m.def("forward_out2_pad", &bottleneck_forward_out2_pad, "Bottleneck block forward");
  m.def("forward_rest", &bottleneck_forward_rest, "Bottleneck block forward");
  m.def("backward_init", &bottleneck_backward_init, "Bottleneck block backward init");
  m.def("backward_grad_out2", &bottleneck_backward_grad_out2, "Bottleneck block backward");
  m.def("backward_grad_out1", &bottleneck_backward_grad_out1, "Bottleneck block backward");
  m.def("backward_grad_out1_mask", &bottleneck_backward_grad_out1_mask, "Bottleneck block backward");
  m.def("backward_grad_out1_halo", &bottleneck_backward_grad_out1_halo, "Bottleneck block backward");
  m.def("backward_grad_out1_halo_corr", &bottleneck_backward_grad_out1_halo_corr, "Bottleneck block backward");
  m.def("backward_wgrad2_pad", &bottleneck_backward_wgrad2_pad, "Bottleneck block backward");
  m.def("backward_wgrad2", &bottleneck_backward_wgrad2, "Bottleneck block backward");
  m.def("backward_wgrad2_halo", &bottleneck_backward_wgrad2_halo, "Bottleneck block backward");
  m.def("backward_wgrad3", &bottleneck_backward_wgrad3, "Bottleneck block backward");
  m.def("backward_wgrad1", &bottleneck_backward_wgrad1, "Bottleneck block backward");
  m.def("backward_rest", &bottleneck_backward_rest, "Bottleneck block backward");
}
