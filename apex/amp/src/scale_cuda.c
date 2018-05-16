#include <THC/THC.h>
#include "scale_kernel.h"

extern THCState *state;

void scale_check_overflow(THCudaTensor *grads,
                          float scale,
                          THCudaByteTensor *overflow_buf) {
  size_t num_elems = THCudaTensor_nElement(state, grads);
  float *d_grads = THCudaTensor_data(state, grads);

  size_t buf_elems = THCudaByteTensor_nElement(state, overflow_buf);
  uint8_t *d_overflow_buf = THCudaByteTensor_data(state, overflow_buf);

  scale_check_overflow_kernel(state, d_grads, num_elems, scale,
                              d_overflow_buf, buf_elems);
}
