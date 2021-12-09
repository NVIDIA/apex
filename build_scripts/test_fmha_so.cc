#include <iostream>
#include "fmhalib.h"

int main() {
  fmhalib_fwd(nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, true, 0, nullptr, 0, false, nullptr, nullptr, nullptr);
  const char *err = fmhalib_get_error();
  if (err != nullptr) std::cout << err << std::endl;
  fmhalib_bwd(nullptr, nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, nullptr, nullptr, nullptr); 
  err = fmhalib_get_error();
  if (err != nullptr) std::cout << err << std::endl;
  return 0;
}
