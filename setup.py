# setup.py
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Optionally drive arch flags via env:
# os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;7.0;7.5;8.0;8.6"

# common nvcc flags
nvcc_flags = [
    "-O2",
    "-gencode=arch=compute_70,code=sm_70",
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-lineinfo"
]

setup(
    name="flashTP_e3nn",
    version="0.1.0",
    python_requires="==3.11.*",
    packages=find_packages(include=["flashTP_e3nn*"], exclude=["example*"]),
    install_requires=[
        "torch>=2.4.1",
        "e3nn",
    ],
    package_data={
        "flashTP_e3nn": [
            "sptp_exp_opt*/*.cpp",
            "sptp_exp_opt*/*.cu",
            "sptp_exp_opt*/*.hpp",
            "sptp_exp_opt*/CMakeLists.txt",
        ]
    },
    ext_modules=[
        # large kernel
        CUDAExtension(
            name="flashTP_e3nn.flashtp_large_kernel",
            sources=[
                "flashTP_e3nn/sptp_exp_opt_large/sptp_exp_opt_large.cpp",
                "flashTP_e3nn/sptp_exp_opt_large/fwd_sptp_linear_v2_shared_exp.cu",
                "flashTP_e3nn/sptp_exp_opt_large/torch_register.cpp",
                "flashTP_e3nn/sptp_exp_opt_large/bwd_sptp_linear_shared_exp.cu",
                "flashTP_e3nn/sptp_exp_opt_large/bwd_bwd_sptp_linear_v2_shared_exp.cu",
                "flashTP_e3nn/sptp_exp_opt_large/fwd_sptp_linear_v2_shared_exp_double.cu",
                "flashTP_e3nn/sptp_exp_opt_large/bwd_sptp_linear_shared_exp_double.cu",
                "flashTP_e3nn/sptp_exp_opt_large/bwd_bwd_sptp_linear_v2_shared_exp_double.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": nvcc_flags,
            },
        ),
        # extcg kernel
        CUDAExtension(
            name="flashTP_e3nn.flashtp_extcg_kernel",
            sources=[
                "flashTP_e3nn/sptp_exp_opt_extcg/sptp_exp_opt.cpp",
                "flashTP_e3nn/sptp_exp_opt_extcg/fwd_sptp_linear_v2_shared_exp.cu",
                "flashTP_e3nn/sptp_exp_opt_extcg/bwd_sptp_linear_shared_exp.cu",
                "flashTP_e3nn/sptp_exp_opt_extcg/bwd_bwd_sptp_linear_v2_shared_exp.cu",
                "flashTP_e3nn/sptp_exp_opt_extcg/fwd_sptp_linear_v2_shared_exp_double.cu",
                "flashTP_e3nn/sptp_exp_opt_extcg/bwd_sptp_linear_shared_exp_double.cu",
                "flashTP_e3nn/sptp_exp_opt_extcg/bwd_bwd_sptp_linear_v2_shared_exp_double.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

