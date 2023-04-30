---
title: NVVM (NVCC & LLVM)
toc: true
date: 2023-04-22 09:33:58
categories:
- Technology
tags:
- NVIDIA
- GPU
- IR
---

NVIDIA's CUDA Compiler (NVCC) is based on the widely used LLVM open source compiler infrastructure. Developers can create or extend programming languages with support for GPU acceleration using the NVIDIA Compiler SDK.

<img src="/img/LLVM_Compiler_structure.jpg" width="50%" />

<!-- more -->

---

## Some links about NVVM.

- [What is cuda llvm compiler](https://developer.nvidia.com/cuda-llvm-compiler)
- [NVVM IR docs](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html)
    - NVVM IR is a compiler IR (intermediate representation) based on the LLVM IR. The NVVM IR is designed to represent GPU compute kernels (for example, CUDA kernels). High-level language front-ends, like the CUDA C compiler front-end, can generate NVVM IR.
- [NVVM IR samples](https://github.com/nvidia-compiler-sdk/nvvmir-samples)
- [LLVM NVPTX backend](https://llvm.org/docs/NVPTXUsage.html)
- [libNVVM API](https://docs.nvidia.com/cuda/libnvvm-api/index.html) 
- [libdevice User's Guide](https://docs.nvidia.com/cuda/libdevice-users-guide/index.html)
    - The libdevice library is an LLVM bitcode library that implements common functions for GPU kernels.

