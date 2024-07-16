module = None

if module is None:
    import os

    from torch.utils.cpp_extension import load

    module = load(
        name="cpp_raconv",
        sources=[f"cpp_source{os.sep}raconv.cpp"],
        extra_cflags=["-std=c++23 -fopenmp -O3"],
    )
