import pathlib


hw_c_list = [
    (8 * 8, 1280),
    (8 * 8, 2560),
    (16 * 16, 640),
    (16 * 16, 1280),
    (16 * 16, 1920),
    (16 * 16, 2560),
    (32 * 32, 320),
    (32 * 32, 640),
    (32 * 32, 960),
    (32 * 32, 1280),
    (32 * 32, 1920),
    (64 * 64, 320),
    (64 * 64, 640),
    (64 * 64, 960),
]


def run():
    src_path = pathlib.Path(__file__).parent.absolute()

    for f in src_path.glob("gn_cuda_inst_*.cu"):
        f.unlink()

    for hw, c in hw_c_list:
        print(f"GN_CUDA_INST_DEFINE({hw}, {c})")
        with open(src_path / f"gn_cuda_inst_{hw}_{c}.cu", "w") as f:
            f.write('#include "gn_cuda_host_template.cuh"\n')
            f.write("\n")
            f.write("\n")
            f.write("namespace group_norm_v2 {\n")
            f.write("\n")
            f.write(f"GN_CUDA_INST_DEFINE({hw}, {c})\n")
            f.write("\n")
            f.write("}  // namespace group_norm_v2\n")

    with open(src_path / "gn_dispatch_hw_c.hpp", "w") as f:
        f.write("#pragma once\n")
        f.write("\n")
        f.write("#define DISPATCH_HW_C(hw, c, HW, C, ...) [&] { \\\n")
        for hw, c in hw_c_list:
            f.write(
                f"    if (hw == {hw} && c == {c}) {{ constexpr int HW = {hw}, C = {c}; return __VA_ARGS__(); }} \\\n"
            )
        f.write(
            '    throw std::invalid_argument("DISPATCH_HW_C " + std::to_string(hw) + " " + std::to_string(c)); \\\n'
        )
        f.write("    }()\n")


if __name__ == "__main__":
    run()
