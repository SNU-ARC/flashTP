# setup.py
import os, glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools.command.develop import develop as _develop
import torch.cuda

def get_arch_list():
    # if defiend by user.
    if env := os.environ.get("CUDA_ARCH_LIST"):
        return env
    print("CUDA_ARCH_LIST not defined, auto-detecting archs from available GPUs...")
    # if not defined get archs from all GPUs
    if torch.cuda.is_available():
        archs = sorted({
            f"{torch.cuda.get_device_capability(i)[0]}{torch.cuda.get_device_capability(i)[1]}"
            for i in range(torch.cuda.device_count())
        })
        if archs:
            print("Detected GPU architectures:", ", ".join(archs))
            return ";".join(archs)
    # fallback
    print("No GPUs detected, defaulting to CUDA_ARCH_LIST=80;90")
    return "80;90"


def make_gencode_flags(arch_list):
    """Given "75 86" returns a flat list of NVCC -gencode flags.
    SASS is emitted for every arch; PTX fallback is added only on the highest arch."""
    archs = [a 
        for a in arch_list.replace(";", " ").replace(",", " ").split()
    ]
    if not archs:
        return []
    flags = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]
    highest = max(archs, key=int)
    flags.append(f"-gencode=arch=compute_{highest},code=compute_{highest}") # PTX fallback
    return flags


CUDA_ARCH_LIST = get_arch_list()
NVCC_GENCODE_FLAGS = make_gencode_flags(CUDA_ARCH_LIST)


# print(NVCC_GENCODE_FLAGS)


# common nvcc flags
nvcc_flags = [
    "-O2",
    "-lineinfo"
] + NVCC_GENCODE_FLAGS


core_kernel = CUDAExtension(
            name="flashTP_e3nn.flashtp_large_core",
            sources=[
                "flashTP_e3nn/sptp_exp_opt_large/sptp_exp_opt_large.cpp",
                "flashTP_e3nn/sptp_exp_opt_large/fwd_sptp_linear_v2_shared_exp.cu",
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
        )
# # large kernel python: uses core_kernel
# large_python_impl = CUDAExtension(
#             name="flashTP_e3nn.flashtp_large_kernel",
#             sources=[
#                 "flashTP_e3nn/sptp_exp_opt_large/python_pybind.cpp",
#                 "flashTP_e3nn/sptp_exp_opt_large/sptp_exp_opt_large.cpp",
#                 "flashTP_e3nn/sptp_exp_opt_large/fwd_sptp_linear_v2_shared_exp.cu",
#                 "flashTP_e3nn/sptp_exp_opt_large/bwd_sptp_linear_shared_exp.cu",
#                 "flashTP_e3nn/sptp_exp_opt_large/bwd_bwd_sptp_linear_v2_shared_exp.cu",
#                 "flashTP_e3nn/sptp_exp_opt_large/fwd_sptp_linear_v2_shared_exp_double.cu",
#                 "flashTP_e3nn/sptp_exp_opt_large/bwd_sptp_linear_shared_exp_double.cu",
#                 "flashTP_e3nn/sptp_exp_opt_large/bwd_bwd_sptp_linear_v2_shared_exp_double.cu",

#             ],
#             extra_compile_args={
#                 "cxx": ["-O2"],
#                 "nvcc": nvcc_flags,
#             }
#         )

# # large kernel lammps: uses core_kernel
# large_lammps_impl = CUDAExtension(
#             name="flashTP_e3nn.flashtp_large_kernel_lammps",
#             sources=[
#                 "flashTP_e3nn/sptp_exp_opt_large/torch_register.cpp",
#                 "flashTP_e3nn/sptp_exp_opt_large/sptp_exp_opt_large.cpp",
#                 "flashTP_e3nn/sptp_exp_opt_large/fwd_sptp_linear_v2_shared_exp.cu",
#                 "flashTP_e3nn/sptp_exp_opt_large/bwd_sptp_linear_shared_exp.cu",
#                 "flashTP_e3nn/sptp_exp_opt_large/bwd_bwd_sptp_linear_v2_shared_exp.cu",
#                 "flashTP_e3nn/sptp_exp_opt_large/fwd_sptp_linear_v2_shared_exp_double.cu",
#                 "flashTP_e3nn/sptp_exp_opt_large/bwd_sptp_linear_shared_exp_double.cu",
#                 "flashTP_e3nn/sptp_exp_opt_large/bwd_bwd_sptp_linear_v2_shared_exp_double.cu",
#             ],
#             extra_compile_args={
#                 "cxx": ["-O2"],
#                 "nvcc": nvcc_flags,
#             },
#         )

# large kernel python: uses core_kernel
large_python_impl = CUDAExtension(
            name="flashTP_e3nn.flashtp_large_kernel",
            sources=[
                "flashTP_e3nn/sptp_exp_opt_large/python_pybind.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": nvcc_flags,
            },
            extra_link_args=["-Wl,-rpath,$ORIGIN"],
        )

# large kernel lammps: uses core_kernel
large_lammps_impl = CUDAExtension(
            name="flashTP_e3nn.flashtp_large_kernel_lammps",
            sources=[
                "flashTP_e3nn/sptp_exp_opt_large/torch_register.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": nvcc_flags,
            },
            extra_link_args=["-Wl,-rpath,$ORIGIN"],
        )

# extcg kernel
extcg_kernel = CUDAExtension(
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
            )

class CustomBuildExt(BuildExtension):
    def build_extensions(self):
        # self.build_lib is the real build output dir
        pkg_build = os.path.join(self.build_lib, "flashTP_e3nn")
        # print("pkg_build", pkg_build)

        for ext in self.extensions:
            # print("ext.name", ext.name)
            if ext.name == "flashTP_e3nn.flashtp_large_core":
                self.core_so = str(self.get_ext_filename(ext.name)).split("/")[-1]
                # print("core.so", self.core_so)
            else:
                # only fix the ones that need to see the core.so
                if ext.name in ["flashTP_e3nn.flashtp_large_kernel", "flashTP_e3nn.flashtp_large_kernel_lammps"]:
                    # print("need core", ext.name)
                    ext.library_dirs.append(str(pkg_build))
                    ext.libraries.append(f":{self.core_so}")
                    
        super().build_extensions()


# class CustomBuildExt(BuildExtension):
#     def build_extensions(self):
#         # self.build_lib is the real build output dir
#         pkg_build = os.path.join(self.build_lib, "flashTP_e3nn")
#         print("pkg_build", pkg_build)

#         core_name = "flashTP_e3nn.flashtp_large_core"
#         core_ext  = next(ext for ext in self.extensions if ext.name == core_name)
#         remaining = [ext for ext in self.extensions if ext is not core_ext]

#         self.extensions = [core_ext]
#         super().build_extensions()
        
#         self.core_so = str(self.get_ext_filename(core_ext.name)).split("/")[-1]
#         print("core.so", self.core_so)

#         self.extensions[:] = remaining

#         for ext in self.extensions:
#             print("ext.name", ext.name)
#             # need to build the core.so first
#             if ext.name == "flashTP_e3nn.flashtp_large_core":
#                 # self.build_extension(ext)
#                 # self.core_so = str(self.get_ext_filename(ext.name)).split("/")[-1]
#                 print("core.so", self.core_so)
#             else:
#                 # only fix the ones that need to see the core.so
#                 if ext.name in ["flashTP_e3nn.flashtp_large_kernel", "flashTP_e3nn.flashtp_large_kernel_lammps"]:
#                     print("need core", ext.name)
#                     ext.library_dirs.append(str(pkg_build))
#                     ext.libraries.append(f":{self.core_so}")
                    
#         super().build_extensions()

class CustomDevelop(_develop):
    def run(self):
        # force the CUDA extensions to build first
        self.run_command("build_ext")
        super().run()

setup(
    name="flashTP_e3nn",
    version="0.1.1",
    python_requires=">=3.10",
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
        core_kernel,
        large_python_impl,
        large_lammps_impl,
        extcg_kernel,
    ],
    cmdclass={"build_ext": CustomBuildExt,
              "develop":  CustomDevelop,},
    options={
        "build_ext": {
            "parallel": 1,
        }
    },
)

