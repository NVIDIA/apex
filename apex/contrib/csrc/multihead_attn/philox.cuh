#pragma once
// Philox CUDA.


namespace {

class Philox {
public:
  __device__ inline Philox(unsigned long long seed,
                           unsigned long long subsequence,
                           unsigned long long offset);

private:
  uint4 counter;
  uint4 output;
  uint2 key;
  unsigned int STATE;
  __device__ inline void incr_n(unsigned long long n);
  __device__ inline void incr();
  __device__ unsigned int mulhilo32(unsigned int a, unsigned int b,
                                    unsigned int *result_high);
  __device__ inline uint4 single_round(uint4 ctr, uint2 key);
  static const unsigned long kPhilox10A = 0x9E3779B9;
  static const unsigned long kPhilox10B = 0xBB67AE85;
  static const unsigned long kPhiloxSA = 0xD2511F53;
  static const unsigned long kPhiloxSB = 0xCD9E8D57;
};
// Inverse of 2^32.
#define M_RAN_INVM32 2.3283064e-10f
__device__ __inline__ float4 uniform4(uint4 x);

} // namespace anonymous
