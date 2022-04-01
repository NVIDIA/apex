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

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(at::MemoryFormat::ChannelsLast), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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

void checkError(cudaError_t code, char const * func, const char *file, const int line, bool abort) {
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


int getFwdConvOutputDim(int tensorDim,
                        int pad,
                        int filterDim,
                        int stride,
                        int dilation) {
    int p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1;
    return (p);
}


// create a cache for plan
std::unordered_map<std::string, cudnn_frontend::ExecutionPlan> plan_cache;


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
run_conv_bias(int64_t* x_dim,
              int64_t* w_dim,
              int64_t* y_dim,
              int64_t* conv_pad,
              int64_t* convstride,
              int64_t* dilation,
              cudnnDataType_t dataType,
              at::Half* devPtrX,
              at::Half* devPtrW,
              at::Half* devPtrB,
              at::Half* devPtrY) {

    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;

    try {
        int convDim = 2;
        float alpha  = 1.0f;
        float beta   = 0.0f;
	int64_t b_dim[] = {1, y_dim[1], 1, 1};

	// Creates the necessary tensor descriptors
	int64_t stride[4];
	generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto xTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, x_dim)
			    .setStrides(4, stride)
			    .setId('x')
			    .setAlignment(16)
			    .setDataType(dataType)
			    .build();
	DEBUG_CUDNN_MSG(log_buf, xTensor.describe());

	generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto wTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, w_dim)
			    .setStrides(4, stride)
			    .setId('w')
			    .setAlignment(16)
			    .setDataType(dataType)
			    .build();
	DEBUG_CUDNN_MSG(log_buf, wTensor.describe());

	generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto afterConvTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, y_dim)
			    .setStrides(4, stride)
			    .setId('c')
			    .setAlignment(16)
			    .setDataType(CUDNN_DATA_FLOAT)
			    .setVirtual()
			    .build();
	DEBUG_CUDNN_MSG(log_buf, afterConvTensor.describe());

	generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto bTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, b_dim)
			    .setStrides(4, stride)
			    .setId('b')
			    .setAlignment(16)
			    .setDataType(dataType)
			    .build();
	DEBUG_CUDNN_MSG(log_buf, bTensor.describe());

	generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto afterBiasTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, y_dim)
			    .setStrides(4, stride)
			    .setId('y')
			    .setAlignment(16)
			    .setDataType(dataType)
			    .build();
	DEBUG_CUDNN_MSG(log_buf, afterBiasTensor.describe());

        // Define the bias operation
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, biasDesc.describe());

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(convDim)
                            .setStrides(convDim, convstride)
                            .setPrePadding(convDim, conv_pad)
                            .setPostPadding(convDim, conv_pad)
                            .setDilation(convDim, dilation)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, convDesc.describe());


        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, conv_op.describe());

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, bias_op.describe());

        // Create an Operation Graph. In this case it is convolution bias activation
        std::array<cudnn_frontend::Operation const*, 2> ops = {&conv_op, &bias_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
          .setHandle(handle_)
          .setOperationGraph(2, ops.data())
          .build();

        // Create string encoding for plan caching
        auto cache_string = getConvFusionString(x_dim, conv_pad, convstride, dilation, w_dim, dataType, opGraph.getTag());
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
        void* data_ptrs[] = {devPtrX, devPtrW, devPtrB, devPtrY};
        int64_t uids[]    = {'x', 'w', 'b', 'y'};
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


void
run_conv_bias_mask_relu(int64_t* x_dim,
                        int64_t* w_dim,
                        int64_t* y_dim,
                        int64_t* conv_pad,
                        int64_t* conv_stride,
                        int64_t* conv_dilation,
                        cudnnDataType_t dataType,
                        at::Half* devPtrX,
                        at::Half* devPtrW,
                        at::Half* devPtrB,
                        int8_t* devPtrM,
                        at::Half* devPtrY) {

    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;

    try {
        int conv_dim = 2;
        float alpha  = 1.0f;
        float beta   = 0.0f;
	int64_t b_dim[] = {1, y_dim[1], 1, 1};

	// Creates the necessary tensor descriptors
	int64_t stride[4];
	generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto xTensor = cudnn_frontend::TensorBuilder()
		 	    .setDim(4, x_dim)
			    .setStrides(4, stride)
			    .setId('x')
			    .setAlignment(16)
			    .setDataType(dataType)
			    .build();
	DEBUG_CUDNN_MSG(log_buf, xTensor.describe());

	generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto wTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, w_dim)
			    .setStrides(4, stride)
			    .setId('w')
			    .setAlignment(16)
			    .setDataType(dataType)
			    .build();
	DEBUG_CUDNN_MSG(log_buf, wTensor.describe());

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto mTensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, y_dim)
                            .setStrides(4, stride)
                            .setId('m')
                            .setAlignment(16)
                            .setDataType(CUDNN_DATA_INT8)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, wTensor.describe());

	generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto afterConvTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, y_dim)
			    .setStrides(4, stride)
			    .setId('c')
			    .setAlignment(16)
			    .setDataType(CUDNN_DATA_FLOAT)
			    .setVirtual()
			    .build();
	DEBUG_CUDNN_MSG(log_buf, afterConvTensor.describe());

	generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto bTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, b_dim)
			    .setStrides(4, stride)
			    .setId('b')
			    .setAlignment(16)
			    .setDataType(dataType)
			    .build();
	DEBUG_CUDNN_MSG(log_buf, bTensor.describe());

	generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto afterBiasTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, y_dim)
			    .setStrides(4, stride)
			    .setId('B')
			    .setAlignment(16)
			    .setDataType(CUDNN_DATA_FLOAT)
			    .setVirtual()
			    .build();
	DEBUG_CUDNN_MSG(log_buf, afterBiasTensor.describe());

        generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto afterMaskTensor = cudnn_frontend::TensorBuilder()
                            .setDim(4, y_dim)
                            .setStrides(4, stride)
                            .setId('M')
                            .setAlignment(16)
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setVirtual()
                            .build();
        DEBUG_CUDNN_MSG(log_buf, afterBiasTensor.describe());

	generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto afterReLUTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, y_dim)
			    .setStrides(4, stride)
			    .setId('y')
			    .setAlignment(16)
			    .setDataType(dataType)
			    .build();
	DEBUG_CUDNN_MSG(log_buf, afterReLUTensor.describe());

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(conv_dim)
                            .setStrides(conv_dim, conv_stride)
                            .setPrePadding(conv_dim, conv_pad)
                            .setPostPadding(conv_dim, conv_pad)
                            .setDilation(conv_dim, conv_dilation)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, convDesc.describe());

        // Define the bias operation
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, biasDesc.describe());

	// Define the mask operation
	auto maskDesc = cudnn_frontend::PointWiseDescBuilder()
			    .setMode(CUDNN_POINTWISE_MUL)
			    .setMathPrecision(CUDNN_DATA_FLOAT)
			    .build();

        // Define the activation operation
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, actDesc.describe());

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, conv_op.describe());

        // Create a Bias Node
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, bias_op.describe());

	// create a Mask Node
	auto mask_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
			   .setxDesc(bias_op.getOutputTensor())
			   .setbDesc(mTensor)
			   .setyDesc(afterMaskTensor)
			   .setpwDesc(maskDesc)
			   .build();
	DEBUG_CUDNN_MSG(log_buf, mask_op.describe());

        // Create an Activation Node
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(mask_op.getOutputTensor())
                          .setyDesc(afterReLUTensor)
                          .setpwDesc(actDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, act_op.describe());

        // Create an Operation Graph. In this case it is convolution bias activation
       std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &bias_op, &mask_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
          .setHandle(handle_)
          .setOperationGraph(4, ops.data())
          .build();

        // Create string encoding for plan caching
        auto cache_string = getConvFusionString(x_dim, conv_pad, conv_stride, conv_dilation, w_dim, dataType, opGraph.getTag());
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
        void* data_ptrs[] = {devPtrX, devPtrW, devPtrB, devPtrM, devPtrY};
        int64_t uids[]    = {'x', 'w', 'b', 'm', 'y'};
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
run_conv_bias_relu(int64_t* x_dim,
                   int64_t* w_dim,
                   int64_t* y_dim,
                   int64_t* conv_pad,
                   int64_t* conv_stride,
                   int64_t* conv_dilation,
                   cudnnDataType_t dataType,
                   at::Half* devPtrX,
                   at::Half* devPtrW,
                   at::Half* devPtrB,
                   at::Half* devPtrY) {

    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;

    try {
        int conv_dim = 2;
        float alpha  = 1.0f;
        float beta   = 0.0f;
	int64_t b_dim[] = {1, y_dim[1], 1, 1};

	// Creates the necessary tensor descriptors
	int64_t stride[4];
	generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto xTensor = cudnn_frontend::TensorBuilder()
		 	    .setDim(4, x_dim)
			    .setStrides(4, stride)
			    .setId('x')
			    .setAlignment(16)
			    .setDataType(dataType)
			    .build();
	DEBUG_CUDNN_MSG(log_buf, xTensor.describe());

	generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto wTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, w_dim)
			    .setStrides(4, stride)
			    .setId('w')
			    .setAlignment(16)
			    .setDataType(dataType)
			    .build();
	DEBUG_CUDNN_MSG(log_buf, wTensor.describe());

	generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto afterConvTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, y_dim)
			    .setStrides(4, stride)
			    .setId('c')
			    .setAlignment(16)
			    .setDataType(CUDNN_DATA_FLOAT)
			    .setVirtual()
			    .build();
	DEBUG_CUDNN_MSG(log_buf, afterConvTensor.describe());

	generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto bTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, b_dim)
			    .setStrides(4, stride)
			    .setId('b')
			    .setAlignment(16)
			    .setDataType(dataType)
			    .build();
	DEBUG_CUDNN_MSG(log_buf, bTensor.describe());

	generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto afterBiasTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, y_dim)
			    .setStrides(4, stride)
			    .setId('B')
			    .setAlignment(16)
			    .setDataType(CUDNN_DATA_FLOAT)
			    .setVirtual()
			    .build();
	DEBUG_CUDNN_MSG(log_buf, afterBiasTensor.describe());

	generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto afterReLUTensor = cudnn_frontend::TensorBuilder()
			    .setDim(4, y_dim)
			    .setStrides(4, stride)
			    .setId('y')
			    .setAlignment(16)
			    .setDataType(dataType)
			    .build();
	DEBUG_CUDNN_MSG(log_buf, afterReLUTensor.describe());

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(conv_dim)
                            .setStrides(conv_dim, conv_stride)
                            .setPrePadding(conv_dim, conv_pad)
                            .setPostPadding(conv_dim, conv_pad)
                            .setDilation(conv_dim, conv_dilation)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, convDesc.describe());

        // Define the bias operation
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, biasDesc.describe());

        // Define the activation operation
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, actDesc.describe());

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                           .setxDesc(xTensor)
                           .setwDesc(wTensor)
                           .setyDesc(afterConvTensor)
                           .setcDesc(convDesc)
                           .setAlpha(alpha)
                           .setBeta(beta)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, conv_op.describe());

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(bTensor)
                           .setyDesc(afterBiasTensor)
                           .setpwDesc(biasDesc)
                           .build();
        DEBUG_CUDNN_MSG(log_buf, bias_op.describe());

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(bias_op.getOutputTensor())
                          .setyDesc(afterReLUTensor)
                          .setpwDesc(actDesc)
                          .build();
        DEBUG_CUDNN_MSG(log_buf, act_op.describe());

        // Create an Operation Graph. In this case it is convolution bias activation
        std::array<cudnn_frontend::Operation const*, 3> ops = {&conv_op, &bias_op, &act_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
          .setHandle(handle_)
          .setOperationGraph(3, ops.data())
          .build();

        // Create string encoding for plan caching
        auto cache_string = getConvFusionString(x_dim, conv_pad, conv_stride, conv_dilation, w_dim, dataType, opGraph.getTag());
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
        void* data_ptrs[] = {devPtrX, devPtrW, devPtrB, devPtrY};
        int64_t uids[]    = {'x', 'w', 'b', 'y'};
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


void
run_drelu_dbias(int64_t* dy_dim,
                cudnnDataType_t dataType,
                at::Half* devPtrDY,
		at::Half* devPtrR,
		at::Half* devPtrDR,
                float* devPtrDB) {

    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;

    try {
	int convDim = 2;
	float alpha = 1.0f;
	float beta = 0.0f;
	int64_t b_dim[] = {1, dy_dim[1], 1, 1};

	// Creates the necessary tensor descriptors
	int64_t stride[4];
	generateStrides(dy_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto dyTensor = cudnn_frontend::TensorBuilder()
		 	 .setDim(4, dy_dim)
		 	 .setStrides(4, stride)
			 .setId('x')
			 .setAlignment(16)
			 .setDataType(dataType)
			 .build();
	DEBUG_CUDNN_MSG(log_buf, dyTensor.describe());
	
	generateStrides(dy_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto rTensor = cudnn_frontend::TensorBuilder()
		 	 .setDim(4, dy_dim)
		 	 .setStrides(4, stride)
			 .setId('r')
			 .setAlignment(16)
			 .setDataType(dataType)
			 .build();
	DEBUG_CUDNN_MSG(log_buf, rTensor.describe());
	
	generateStrides(dy_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto inActGradTensor = cudnn_frontend::TensorBuilder()
		 	 .setDim(4, dy_dim)
		 	 .setStrides(4, stride)
			 .setId('R')
			 .setAlignment(16)
			 .setDataType(dataType)
			 .build();
	DEBUG_CUDNN_MSG(log_buf, inActGradTensor.describe());

        generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto biasGradTensor = cudnn_frontend::TensorBuilder()
                         .setDim(4, b_dim)
                         .setStrides(4, stride)
                         .setId('y')
                         .setAlignment(16)
                         .setDataType(CUDNN_DATA_FLOAT)
                         .build();
        DEBUG_CUDNN_MSG(log_buf, biasGradTensor.describe());

	// Define the activation backward operation
	auto actDesc = cudnn_frontend::PointWiseDescBuilder()
	  .setMode(CUDNN_POINTWISE_RELU_BWD)
	  .setMathPrecision(CUDNN_DATA_FLOAT)
	  .build();
	DEBUG_CUDNN_MSG(log_buf, actDesc.describe());

	// Define the bias backward operation
	auto biasDesc = cudnn_frontend::ReductionDescBuilder()
          .setMathPrecision(CUDNN_DATA_FLOAT)
	  .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
	  .build();
	DEBUG_CUDNN_MSG(log_buf, biasDesc.describe());

	// Create an relu backward Node
	auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
	  .setdyDesc(dyTensor)
	  .setxDesc(rTensor)
	  .setdxDesc(inActGradTensor)
	  .setpwDesc(actDesc)
	  .build();
	DEBUG_CUDNN_MSG(log_buf, act_op.describe());

	// Create bias node
	auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
	  .setxDesc(inActGradTensor)
	  .setyDesc(biasGradTensor)
	  .setreductionDesc(biasDesc)
	  .build();
	DEBUG_CUDNN_MSG(log_buf, bias_op.describe());

	// Create an Operation Graph. In this case it is bias only
	std::array<cudnn_frontend::Operation const*, 2> ops = {&act_op, &bias_op};

	auto opGraph = cudnn_frontend::OperationGraphBuilder()
	  .setHandle(handle_)
	  .setOperationGraph(ops.size(), ops.data())
	  .build();

	// Create string encoding for plan caching
	// creating unique dummy values
	int64_t pad_dummy[] = {20, 20};
	int64_t stride_dummy[] = {20, 20};
	int64_t dilation_dummy[] = {20, 20};
	auto cache_string = getConvFusionString(dy_dim, pad_dummy, stride_dummy, dilation_dummy, b_dim, dataType, opGraph.getTag());
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
	void* data_ptrs[] = {devPtrDY, devPtrR, devPtrDR, devPtrDB};
	int64_t uids[]    = {'x', 'r', 'R', 'y'};
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


void
run_dconv_drelu_dbias(int64_t* x_dim,
		      int64_t* w_dim,
                      int64_t* y_dim,
		      int64_t* pad,
		      int64_t* convstride,
		      int64_t* dilation,
                      cudnnDataType_t dataType,
                      at::Half* devPtrX,
		      at::Half* devPtrW,
                      at::Half* devPtrR,
                      at::Half* devPtrRg,
                      float* devPtrY) {
    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;
    try {
	int convDim = 2;
	float alpha = 1.0f;
	float beta = 0.0f;
	int64_t b_dim[] = {1, x_dim[1], 1, 1};

	int64_t stride[4];
	generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto outConvGradTensor = cudnn_frontend::TensorBuilder()
			 .setDim(4, y_dim)
			 .setStrides(4, stride)
			 .setId('x')
			 .setAlignment(16)
			 .setDataType(dataType)
			 .build();
	DEBUG_CUDNN_MSG(log_buf, outConvGradTensor.describe());

        generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto wTensor = cudnn_frontend::TensorBuilder()
                         .setDim(4, w_dim)
                         .setStrides(4, stride)
                         .setId('w')
                         .setAlignment(16)
                         .setDataType(dataType)
                         .build();
        DEBUG_CUDNN_MSG(log_buf, wTensor.describe());

        generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto inConvGradTensor = cudnn_frontend::TensorBuilder()
                         .setDim(4, x_dim)
                         .setStrides(4, stride)
                         .setId('A')
                         .setAlignment(16)
                         .setDataType(CUDNN_DATA_FLOAT)
			 .setVirtual()
                         .build();
        DEBUG_CUDNN_MSG(log_buf, inConvGradTensor.describe());

        generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto rTensor = cudnn_frontend::TensorBuilder()
                         .setDim(4, x_dim)
                         .setStrides(4, stride)
                         .setId('r')
                         .setAlignment(16)
                         .setDataType(dataType)
                         .build();
        DEBUG_CUDNN_MSG(log_buf, rTensor.describe());

        generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto inReLUGradTensor = cudnn_frontend::TensorBuilder()
                         .setDim(4, x_dim)
                         .setStrides(4, stride)
                         .setId('R')
                         .setAlignment(16)
                         .setDataType(dataType)
                         .build();
        DEBUG_CUDNN_MSG(log_buf, inReLUGradTensor.describe());

        generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
        auto inBiasGradTensor = cudnn_frontend::TensorBuilder()
                         .setDim(4, b_dim)
                         .setStrides(4, stride)
                         .setId('y')
                         .setAlignment(16)
                         .setDataType(CUDNN_DATA_FLOAT)
                         .build();
        DEBUG_CUDNN_MSG(log_buf, inBiasGradTensor.describe());

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

	// Define the bias backward operation
	auto biasDesc = cudnn_frontend::ReductionDescBuilder()
          .setMathPrecision(CUDNN_DATA_FLOAT)
	  .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
	  .build();
	DEBUG_CUDNN_MSG(log_buf, biasDesc.describe());

	// Create a convolution Node
	auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
	  .setdyDesc(outConvGradTensor)
	  .setwDesc(wTensor)
	  .setdxDesc(inConvGradTensor)
	  .setcDesc(convDesc)
	  .setAlpha(alpha)
	  .setBeta(beta)
	  .build();
	DEBUG_CUDNN_MSG(log_buf, conv_op.describe());

	// Create an relu backward Node
	auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
	  .setdyDesc(inConvGradTensor)
	  .setxDesc(rTensor)
	  .setdxDesc(inReLUGradTensor)
	  .setpwDesc(actDesc)
	  .build();
	DEBUG_CUDNN_MSG(log_buf, act_op.describe());

	// Create bias node
	auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
	  .setxDesc(inReLUGradTensor)
	  .setyDesc(inBiasGradTensor)
	  .setreductionDesc(biasDesc)
	  .build();
	DEBUG_CUDNN_MSG(log_buf, bias_op.describe());

	// Create an Operation Graph. In this case it is bias only
	std::array<cudnn_frontend::Operation const*, 3> ops = {&conv_op, &act_op, &bias_op};

	auto opGraph = cudnn_frontend::OperationGraphBuilder()
	  .setHandle(handle_)
	  .setOperationGraph(ops.size(), ops.data())
	  .build();

	// Create string encoding for plan caching
	auto cache_string = getConvFusionString(x_dim, pad, convstride, dilation, w_dim, dataType, opGraph.getTag());
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
	void* data_ptrs[] = {devPtrX, devPtrW, devPtrR, devPtrRg, devPtrY};
	int64_t uids[]    = {'x', 'w', 'r', 'R', 'y'};
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
run_dconv(int64_t* x_dim,
          int64_t* w_dim,
          int64_t* y_dim,
          int64_t* conv_pad,
          int64_t* conv_stride,
          int64_t* conv_dilation,
          cudnnDataType_t dataType,
          at::Half* devPtrX,
          at::Half* devPtrW,
          at::Half* devPtrY,
          cudnnBackendDescriptorType_t mode) {

    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;

    try {
        int conv_dim = 2;
        float alpha  = 1.0f;
        float beta   = 0.0f;

        // Define the convolution problem
	int64_t stride[4];
	generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto xTensor = cudnn_frontend::TensorBuilder()
			  .setDim(4, x_dim)
			  .setStrides(4, stride)
			  .setId('x')
			  .setAlignment(16)
			  .setDataType(dataType)
			  .build();
	DEBUG_CUDNN_MSG(log_buf, xTensor.describe());

	generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto wTensor = cudnn_frontend::TensorBuilder()
			  .setDim(4, w_dim)
			  .setStrides(4, stride)
			  .setId('w')
			  .setAlignment(16)
			  .setDataType(dataType)
			  .build();
	DEBUG_CUDNN_MSG(log_buf, wTensor.describe());

	generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto yTensor = cudnn_frontend::TensorBuilder()
			  .setDim(4, y_dim)
			  .setStrides(4, stride)
			  .setId('y')
			  .setAlignment(16)
			  .setDataType(dataType)
			  .build();
	DEBUG_CUDNN_MSG(log_buf, yTensor.describe());


        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
                            .setDataType(CUDNN_DATA_FLOAT)
                            .setMathMode(CUDNN_CROSS_CORRELATION)
                            .setNDims(conv_dim)
                            .setStrides(conv_dim, conv_stride)
                            .setPrePadding(conv_dim, conv_pad)
                            .setPostPadding(conv_dim, conv_pad)
                            .setDilation(conv_dim, conv_dilation)
                            .build();
        DEBUG_CUDNN_MSG(log_buf, convDesc.describe());

        // Create a convolution node
        // mode should be one of following
        // CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
        // CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR
        auto conv_op_builder = cudnn_frontend::OperationBuilder(mode);
        if (mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
          conv_op_builder.setdxDesc(xTensor)
            .setwDesc(wTensor)
            .setdyDesc(yTensor)
            .setcDesc(convDesc);
        }
        else {
          conv_op_builder.setxDesc(xTensor)
            .setdwDesc(wTensor)
            .setdyDesc(yTensor)
            .setcDesc(convDesc);
        }
        auto conv_op = conv_op_builder
		         .setAlpha(alpha)
			 .setBeta(beta)
			 .build();
        DEBUG_CUDNN_MSG(log_buf, conv_op.describe());

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 1> ops = {&conv_op};

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
          .setHandle(handle_)
          .setOperationGraph(ops.size(), ops.data())
          .build();

        // Create string encoding for plan caching
        auto cache_string = getConvFusionString(x_dim, conv_pad, conv_stride, conv_dilation, w_dim, dataType, opGraph.getTag());
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
        void* data_ptrs[] = {devPtrX, devPtrW, devPtrY};
        int64_t uids[]    = {'x', 'w', 'y'};
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
run_dbias(int64_t* x_dim,
          cudnnDataType_t dataType,
          at::Half* devPtrX,
          float* devPtrY) {
    cudnnHandle_t handle_ = torch::native::getCudnnHandle();
    std::stringstream log_buf;
    try {
	int convDim = 2;
	int64_t b_dim[] = {1, x_dim[1], 1, 1};

	int64_t stride[4];
	generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto xTensor = cudnn_frontend::TensorBuilder()
			 .setDim(4, x_dim)
			 .setStrides(4, stride)
			 .setId('x')
			 .setAlignment(16)
			 .setDataType(dataType)
			 .build();
	DEBUG_CUDNN_MSG(log_buf, xTensor.describe());

	generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
	auto yTensor = cudnn_frontend::TensorBuilder()
			 .setDim(4, b_dim)
			 .setStrides(4, stride)
			 .setId('y')
			 .setAlignment(16)
			 .setDataType(CUDNN_DATA_FLOAT)
			 .build();
	DEBUG_CUDNN_MSG(log_buf, yTensor.describe());

	// Define the bias backward operation
	auto biasDesc = cudnn_frontend::ReductionDescBuilder()
          .setMathPrecision(CUDNN_DATA_FLOAT)
	  .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
	  .build();
	DEBUG_CUDNN_MSG(log_buf, biasDesc.describe());

	// Create bias node
	auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
	  .setxDesc(xTensor)
	  .setyDesc(yTensor)
	  .setreductionDesc(biasDesc)
	  .build();
	DEBUG_CUDNN_MSG(log_buf, bias_op.describe());

	// Create an Operation Graph. In this case it is bias only
	std::array<cudnn_frontend::Operation const*, 1> ops = {&bias_op};

	auto opGraph = cudnn_frontend::OperationGraphBuilder()
	  .setHandle(handle_)
	  .setOperationGraph(ops.size(), ops.data())
	  .build();

	// Create string encoding for plan caching
	int64_t pad_dummy[] = {10, 10};
	int64_t stride_dummy[] = {10, 10};
	int64_t dilation_dummy[] = {10, 10};
	auto cache_string = getConvFusionString(x_dim, pad_dummy, stride_dummy, dilation_dummy, b_dim, dataType, opGraph.getTag());
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
	void* data_ptrs[] = {devPtrX, devPtrY};
	int64_t uids[]    = {'x', 'y'};
	auto variantPack  = cudnn_frontend::VariantPackBuilder()
	  .setWorkspacePointer(workspace_ptr)
	  .setDataPointers(2, data_ptrs)
	  .setUids(2, uids)
	  .build();
	DEBUG_CUDNN_MSG(log_buf, "variantPack " << variantPack.describe());
	cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
	checkCudnnErr(status);
	cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
    } catch (cudnn_frontend::cudnnException e) {
        std::cout << log_buf.str() << "[ERROR] Exception " << e.what() << std::endl;
    }

}


std::vector<at::Tensor> conv_bias_mask_relu_forward(std::vector<at::Tensor> inputs, int64_t padding, int64_t stride) {
  std::cout << std::fixed;

  // create output vector
  std::vector<at::Tensor> outputs;
  auto output_format = at::MemoryFormat::ChannelsLast;

  // setup dimensions
  int64_t x_dim[]        = {0, 0, 0, 0};
  int64_t w_dim[]        = {0, 0, 0, 0};

  // All dim calculation after this order of n,c,h,w
  int axis[] = {0, 1, 2, 3};
  for (int dim = 0; dim < 4; dim++) {
    x_dim[dim] = inputs[0].size(axis[dim]);
    w_dim[dim] = inputs[1].size(axis[dim]);
  }

  // output dim in n,c,h,w used by backend
  int64_t y_dim[]     = {0, 0, 0, 0};

  // use these fixed values
  int64_t conv_pad[]        = {padding, padding};
  int64_t conv_stride[]     = {stride, stride};
  int64_t conv_dilation[]   = {1, 1};

  // compute output from pad/stride/dilation
  y_dim[0] = x_dim[0];
  y_dim[1] = w_dim[0];
  for (int dim = 0; dim < 2; dim++) {
    y_dim[dim + 2] = getFwdConvOutputDim(x_dim[dim + 2], conv_pad[dim], w_dim[dim + 2], conv_stride[dim], conv_dilation[dim]);
  }

  // run
  at::Half* x = inputs[0].data_ptr<at::Half>();
  at::Half* w = inputs[1].data_ptr<at::Half>();
  at::Half* b = inputs[2].data_ptr<at::Half>();
  int8_t* m = inputs[3].data_ptr<int8_t>();
  auto out = at::empty(y_dim, inputs[0].type(), output_format);
  at::Half* y = out.data_ptr<at::Half>();

  run_conv_bias_mask_relu(x_dim,
		          w_dim,
      		          y_dim,
		          conv_pad,
		          conv_stride,
		          conv_dilation,
		          CUDNN_DATA_HALF,
		          x,
		          w,
		          b,
		          m,
		          y);

  DEBUG_MSG("[DEBUG] conv-bias-mask-relu : " << y.to(at::kFloat).sum().item<float>());

  outputs.push_back(out);

  return outputs;
}


std::vector<at::Tensor> conv_bias_relu_forward(std::vector<at::Tensor> inputs, int64_t padding, int64_t stride) {
  std::cout << std::fixed;

  // create output vector
  std::vector<at::Tensor> outputs;
  auto output_format = at::MemoryFormat::ChannelsLast;

  // setup dimensions
  int64_t x_dim[]        = {0, 0, 0, 0};
  int64_t w_dim[]        = {0, 0, 0, 0};

  // All dim calculation after this order of n,c,h,w
  int axis[] = {0, 1, 2, 3};
  for (int dim = 0; dim < 4; dim++) {
    x_dim[dim] = inputs[0].size(axis[dim]);
    w_dim[dim] = inputs[1].size(axis[dim]);
  }

  // output dim in n,c,h,w used by backend
  int64_t y_dim[]     = {0, 0, 0, 0};

  // use these fixed values
  int64_t conv_pad[]        = {padding, padding};
  int64_t conv_stride[]     = {stride, stride};
  int64_t conv_dilation[]   = {1, 1};

  // compute output from pad/stride/dilation
  y_dim[0] = x_dim[0];
  y_dim[1] = w_dim[0];
  for (int dim = 0; dim < 2; dim++) {
    y_dim[dim + 2] = getFwdConvOutputDim(x_dim[dim + 2], conv_pad[dim], w_dim[dim + 2], conv_stride[dim], conv_dilation[dim]);
  }

  // run
  at::Half* x = inputs[0].data_ptr<at::Half>();
  at::Half* w = inputs[1].data_ptr<at::Half>();
  at::Half* b = inputs[2].data_ptr<at::Half>();
  auto out = at::empty(y_dim, inputs[0].type(), output_format);
  at::Half* y = out.data_ptr<at::Half>();

  run_conv_bias_relu(x_dim,
		     w_dim,
		     y_dim,
		     conv_pad,
		     conv_stride,
		     conv_dilation,
		     CUDNN_DATA_HALF,
		     x,
		     w,
		     b,
		     y);

  DEBUG_MSG("[DEBUG] conv-bias-relu : " << y.to(at::kFloat).sum().item<float>());

  outputs.push_back(out);

  return outputs;
}


std::vector<at::Tensor> conv_bias_relu_backward(std::vector<at::Tensor> inputs, int64_t padding, int64_t stride) {
  bool requires_grad = inputs[0].requires_grad();

  for (int i = 0; i <= 3; i++) {
    CHECK_INPUT(inputs[i]);
  }

  std::cout << std::fixed;

  // create output vector
  std::vector<at::Tensor> outputs;
  auto output_format = at::MemoryFormat::ChannelsLast;

  // setup dimensions
  int64_t x_dim[]        = {0, 0, 0, 0};
  int64_t w_dim[]	 = {0, 0, 0, 0};
  int64_t y_dim[]	 = {0, 0, 0, 0};

  // All dim calculation after this order of n,c,h,w
  int axis[] = {0, 1, 2, 3};
  for (int dim = 0; dim < 4; dim++) {
    x_dim[dim] = inputs[0].size(axis[dim]);
    w_dim[dim] = inputs[1].size(axis[dim]);
    y_dim[dim] = inputs[3].size(axis[dim]);
  }
  
  int64_t b_dim[]       = {1, y_dim[1], 1, 1};

  int64_t conv_pad[]        = {padding, padding};
  int64_t conv_stride[]     = {stride, stride};
  int64_t conv_dilation[]   = {1, 1};

  // run
  // drelu-dbias
  at::Half* dy = inputs[3].data_ptr<at::Half>();
  at::Half* r = inputs[2].data_ptr<at::Half>();
  auto drelu = at::empty_like(inputs[2]);
  at::Half* dr = drelu.data_ptr<at::Half>();
  auto options = at::TensorOptions().dtype(at::kFloat).layout(inputs[0].layout()).device(inputs[0].device()).requires_grad(false);
  auto bgrad = at::empty(b_dim, options, output_format);
  float* db = bgrad.data_ptr<float>();
  run_drelu_dbias(y_dim,
		  CUDNN_DATA_HALF,
		  dy,
		  r,
		  dr,
		  db);

  // conv wgrad
  at::Half* x = inputs[0].data_ptr<at::Half>();
  auto wgrad = at::empty_like(inputs[1]);
  at::Half* dw = wgrad.data_ptr<at::Half>(); 
  run_dconv(x_dim,
	    w_dim,
	    y_dim,
	    conv_pad,
	    conv_stride,
	    conv_dilation,
	    CUDNN_DATA_HALF,
	    x,
	    dw,
	    dr,
	    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);

  // conv dgrad
  at::Half* w = inputs[1].data_ptr<at::Half>();
  auto dgrad = at::empty_like(inputs[0]);
  at::Half* dx = dgrad.data_ptr<at::Half>();
  run_dconv(x_dim,
	    w_dim,
	    y_dim,
	    conv_pad,
	    conv_stride,
	    conv_dilation,
	    CUDNN_DATA_HALF,
	    dx,
	    w,
	    dr,
	    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);

  outputs.push_back(dgrad);
  outputs.push_back(wgrad);
  outputs.push_back(bgrad);

  return outputs;

}

std::vector<at::Tensor> conv_bias_forward(std::vector<at::Tensor> inputs, int64_t padding, int64_t stride) {
  std::cout << std::fixed;

  // create output vector
  std::vector<at::Tensor> outputs;
  auto output_format = at::MemoryFormat::ChannelsLast;

  // setup dimensions
  int64_t x_dim[]        = {0, 0, 0, 0};
  int64_t w_dim[]        = {0, 0, 0, 0};

  // All dim calculation after this order of n,c,h,w
  int axis[] = {0, 1, 2, 3};
  for (int dim = 0; dim < 4; dim++) {
    x_dim[dim] = inputs[0].size(axis[dim]);
    w_dim[dim] = inputs[1].size(axis[dim]);
  }

  // output dim in n,c,h,w used by backend
  int64_t y_dim[]     = {0, 0, 0, 0};

  // use these fixed values
  int64_t conv_pad[]        = {padding, padding};
  int64_t conv_stride[]     = {stride, stride};
  int64_t conv_dilation[]   = {1, 1};

  // compute output from pad/stride/dilation
  y_dim[0] = x_dim[0];
  y_dim[1] = w_dim[0];
  for (int dim = 0; dim < 2; dim++) {
    y_dim[dim + 2] = getFwdConvOutputDim(x_dim[dim + 2], conv_pad[dim], w_dim[dim + 2], conv_stride[dim], conv_dilation[dim]);
  }

  // run
  at::Half* x = inputs[0].data_ptr<at::Half>();
  at::Half* w = inputs[1].data_ptr<at::Half>();
  at::Half* b = inputs[2].data_ptr<at::Half>();
  auto out = at::empty(y_dim, inputs[0].type(), output_format);
  at::Half* y = out.data_ptr<at::Half>();

  run_conv_bias(x_dim,
	        w_dim,
	        y_dim,
	        conv_pad,
	        conv_stride,
	        conv_dilation,
	        CUDNN_DATA_HALF,
	        x,
	        w,
	        b,
	        y);

  DEBUG_MSG("[DEBUG] conv-bias : " << y.to(at::kFloat).sum().item<float>());

  outputs.push_back(out);

  return outputs;
}


std::vector<at::Tensor> conv_bias_backward(std::vector<at::Tensor> inputs, int64_t padding, int64_t stride) {
  bool requires_grad = inputs[0].requires_grad();

  for (int i = 0; i <= 2; i++) {
    CHECK_INPUT(inputs[i]);
  }

  std::cout << std::fixed;

  // create output vector
  std::vector<at::Tensor> outputs;
  auto output_format = at::MemoryFormat::ChannelsLast;

  // setup dimensions
  int64_t x_dim[]        = {0, 0, 0, 0};
  int64_t w_dim[]	 = {0, 0, 0, 0};
  int64_t y_dim[]	 = {0, 0, 0, 0};

  // All dim calculation after this order of n,c,h,w
  int axis[] = {0, 1, 2, 3};
  for (int dim = 0; dim < 4; dim++) {
    x_dim[dim] = inputs[0].size(axis[dim]);
    w_dim[dim] = inputs[1].size(axis[dim]);
    y_dim[dim] = inputs[2].size(axis[dim]);
  }
  
  int64_t b_dim[]       = {1, y_dim[1], 1, 1};

  int64_t conv_pad[]        = {padding, padding};
  int64_t conv_stride[]     = {stride, stride};
  int64_t conv_dilation[]   = {1, 1};

  // run
  // dbias
  at::Half* dy = inputs[2].data_ptr<at::Half>();
  auto options = at::TensorOptions().dtype(at::kFloat).layout(inputs[0].layout()).device(inputs[0].device()).requires_grad(false);
  auto bgrad = at::empty(b_dim, options, output_format);
  float* db = bgrad.data_ptr<float>();
  run_dbias(y_dim,
            CUDNN_DATA_HALF,
            dy,
            db); 
  
  // conv wgrad
  at::Half* x = inputs[0].data_ptr<at::Half>();
  auto wgrad = at::empty_like(inputs[1]);
  at::Half* dw = wgrad.data_ptr<at::Half>(); 
  run_dconv(x_dim,
	    w_dim,
	    y_dim,
	    conv_pad,
	    conv_stride,
	    conv_dilation,
	    CUDNN_DATA_HALF,
	    x,
	    dw,
	    dy,
	    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);

  // conv dgrad
  at::Half* w = inputs[1].data_ptr<at::Half>();
  auto dgrad = at::empty_like(inputs[0]);
  at::Half* dx = dgrad.data_ptr<at::Half>();
  run_dconv(x_dim,
	    w_dim,
	    y_dim,
	    conv_pad,
	    conv_stride,
	    conv_dilation,
	    CUDNN_DATA_HALF,
	    dx,
	    w,
	    dy,
	    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);

  outputs.push_back(dgrad);
  outputs.push_back(wgrad);
  outputs.push_back(bgrad);

  return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_bias_relu_forward, "Fused Conv-Bias-ReLU forward");
  m.def("backward", &conv_bias_relu_backward, "Fused Conv-Bias-ReLU backward");
  m.def("forward_no_relu", &conv_bias_forward, "Fused Conv-Bias forward");
  m.def("backward_no_relu", &conv_bias_backward, "Fused Conv-Bias backward");
  m.def("forward_mask", &conv_bias_mask_relu_forward, "Fused Conv-Bias-Mask-ReLU forward");
}

