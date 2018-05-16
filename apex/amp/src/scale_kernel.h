#ifndef SCALE_KERNEL_H
#define SCALE_KERNEL_H

#include <THC/THC.h>

#ifdef __cplusplus
extern "C" {
#endif

void scale_check_overflow_kernel(THCState *state,
                                 float *d_grads, size_t n, float scale,
                                 uint8_t *d_buf, size_t buf_n);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SCALE_KERNEL_H
