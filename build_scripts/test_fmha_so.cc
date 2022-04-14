#include <iostream>
#include "fmhalib.h"

void PrintError() {
  const char *err = fmhalib_error();
  if (err) std::cout << err << std::endl;
}

void PrintErrorCxx() {
  const char *err = fmhalib::error();
  if (err) std::cout << err << std::endl;
}

int main() {
  fmhalib_random_seed(1000);
  PrintError();

  fmhalib::random_seed(1000);
  PrintErrorCxx();

  int seq_len = 512;
  uint64_t rnd_seed;
  int64_t *offset_ptr;
  uint64_t rnd_offset;
  bool is_device_rnd;

  auto inc = fmhalib_random_increment(seq_len);  
  fmhalib_random_state(inc, &rnd_seed, &offset_ptr, &rnd_offset, &is_device_rnd); 
  PrintError();

  inc = fmhalib_random_increment(seq_len);
  fmhalib::random_state(inc, &rnd_seed, &offset_ptr, &rnd_offset, &is_device_rnd);
  PrintErrorCxx();

  fmhalib_fwd(nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, true, 0, nullptr, 0, false, nullptr, nullptr, nullptr);
  PrintError();

  fmhalib_fwd_nl(nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, true, 0, nullptr, 0, false, nullptr, nullptr, nullptr);
  PrintError();

  fmhalib_bwd(nullptr, nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, nullptr, nullptr, nullptr); 
  PrintError();

  fmhalib_bwd_nl(nullptr, nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, nullptr, nullptr, nullptr, nullptr);
  PrintError();

  // The following codes show the dynload style
  fmhalib::fwd(nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, true, 0, nullptr, 0, false, nullptr, nullptr, nullptr);
  PrintErrorCxx();

  fmhalib::fwd_nl(nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, true, 0, nullptr, 0, false, nullptr, nullptr, nullptr);
  PrintErrorCxx();

  fmhalib::bwd(nullptr, nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, nullptr, nullptr, nullptr);
  PrintErrorCxx();

  fmhalib::bwd_nl(nullptr, nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, nullptr, nullptr, nullptr, nullptr);
  PrintErrorCxx();

  std::cout << fmhalib_seq_len(200) << std::endl;
  std::cout << fmhalib::seq_len(200) << std::endl;
  
  return 0;
}
