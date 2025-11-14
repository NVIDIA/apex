#pragma once

#define DISPATCH_HW_C(hw, c, HW, C, ...)                                                          \
  [&] {                                                                                           \
    if (hw == 64 && c == 1280) {                                                                  \
      constexpr int HW = 64, C = 1280;                                                            \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 64 && c == 2560) {                                                                  \
      constexpr int HW = 64, C = 2560;                                                            \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 256 && c == 640) {                                                                  \
      constexpr int HW = 256, C = 640;                                                            \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 256 && c == 1280) {                                                                 \
      constexpr int HW = 256, C = 1280;                                                           \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 256 && c == 1920) {                                                                 \
      constexpr int HW = 256, C = 1920;                                                           \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 256 && c == 2560) {                                                                 \
      constexpr int HW = 256, C = 2560;                                                           \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 1024 && c == 320) {                                                                 \
      constexpr int HW = 1024, C = 320;                                                           \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 1024 && c == 640) {                                                                 \
      constexpr int HW = 1024, C = 640;                                                           \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 1024 && c == 960) {                                                                 \
      constexpr int HW = 1024, C = 960;                                                           \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 1024 && c == 1280) {                                                                \
      constexpr int HW = 1024, C = 1280;                                                          \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 1024 && c == 1920) {                                                                \
      constexpr int HW = 1024, C = 1920;                                                          \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 4096 && c == 320) {                                                                 \
      constexpr int HW = 4096, C = 320;                                                           \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 4096 && c == 640) {                                                                 \
      constexpr int HW = 4096, C = 640;                                                           \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    if (hw == 4096 && c == 960) {                                                                 \
      constexpr int HW = 4096, C = 960;                                                           \
      return __VA_ARGS__();                                                                       \
    }                                                                                             \
    throw std::invalid_argument("DISPATCH_HW_C " + std::to_string(hw) + " " + std::to_string(c)); \
  }()
