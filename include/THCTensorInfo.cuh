#ifndef THC_TENSOR_INFO_INC
#define THC_TENSOR_INFO_INC

#include <cuda.h>
#include <cuda_fp16.h>
#include <assert.h>

// Maximum number of dimensions allowed for cutorch
#define MAX_CUTORCH_DIMS 10

// Warning string for tensor arguments that are too large or have too
// many dimensions
#define CUTORCH_STR(X) #X
#define CUTORCH_DIM_WARNING "tensor too large or too many (>" \
  CUTORCH_STR(MAX_CUTORCH_DIMS) ") dimensions"

enum float_types { FLOAT = 0 , HALF = 1, DOUBLE = 2 };

// CUDA kernel argument that defines tensor layout
template <typename T, typename IndexType>
struct TensorInfo {
  
  TensorInfo(T* p,
             int dim,
             IndexType sz[MAX_CUTORCH_DIMS],
             IndexType st[MAX_CUTORCH_DIMS]);
  
  TensorInfo(T* p,
	     int dim,
	     IndexType sz[MAX_CUTORCH_DIMS],
	     IndexType st[MAX_CUTORCH_DIMS],
	     float_types type);

  //Good way to cast from another format
  //template <TensorInfo<typename T2, typename I2> >
  //TensorInfo(TensorInfo<T2, I2> &tinfo_in){
  //  data = reinterpret_cast<T*>(tinfo_in.data);
  //}
  
  T* data;
  IndexType sizes[MAX_CUTORCH_DIMS];
  IndexType strides[MAX_CUTORCH_DIMS];
  int dims;
  float_types type;
};

//Expand our combinations as convenient typedefs
typedef TensorInfo<half, int> t_hi;
typedef TensorInfo<half, long> t_hl;
typedef TensorInfo<float, int> t_fi;
typedef TensorInfo<float, long> t_fl;


template <typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo(T* p,
                                     int dim,
                                     IndexType sz[MAX_CUTORCH_DIMS],
                                     IndexType st[MAX_CUTORCH_DIMS]) {
  data = p;
  dims = dim;
  assert(dims > 0 && dims < MAX_CUTORCH_DIMS);

  for (int i = 0; i < dim; ++i) {
    sizes[i] = sz[i];
    strides[i] = st[i];
  }
}

template<typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo(T* p,
	   int dim,
	   IndexType sz[MAX_CUTORCH_DIMS],
	   IndexType st[MAX_CUTORCH_DIMS],
	   float_types _type){
  data = p;
  dims = dim;
  assert(dims > 0 && dims < MAX_CUTORCH_DIMS);

  for (int i = 0; i < dim; ++i) {
    sizes[i] = sz[i];
    strides[i] = st[i];
  }
  type=_type;
}



// Translate a linear index for the apply to a T* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename T, typename IndexType, int Dims>
struct IndexToOffset {
  static __forceinline__ __host__ __device__ IndexType get(
					   IndexType linearId,
					   const TensorInfo<T, IndexType>& info) {
    IndexType offset = 0;
    
    // Use static dims
    for (int i = Dims - 1; i > 0; --i) {
      for (int i = Dims - 1; i > 0; --i) {
	offset += linearId % info.sizes[i] * info.strides[i];
	linearId /= info.sizes[i];
      }
      
      offset += linearId * info.strides[0];
      return offset;
    }
  }
};
  

  
// For contiguous tensors, the offset = index
template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -2> {
  static __forceinline__ __host__ __device__ IndexType
    get(IndexType linearId, const TensorInfo<T, IndexType>& info) {
    return linearId;
  }
};

template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -1> {
  static __forceinline__ __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<T, IndexType>& info) {

    IndexType offset = 0;

    // Use dynamic dims
    for (int i = info.dims - 1; i >= 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      linearId /= info.sizes[i];
    }

    return offset;
  }
};

#endif // THC_TENSOR_INFO_INC
