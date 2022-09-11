---
layout: default
title: V.Explore GEMM and GPGPU-SIM (optional)
nav_order: 7
---

## V Explore GEMM and GPGPU-SIM `optional`

{: .highlight}
> This part of Lab is optional

### V.1 Build [GPGPU-SIM](http://www.gpgpu-sim.org/)

{: .highlight}
> `dev` TODO using docker

GPGPU-SIM is a simulator for CUDA program. GPGPU-SIM is a little outdated from GEM5. But it is still acknowledged by academic field.

- You may find these tutorials helpful
  
  - [install GPGPU-SIM](https://github.com/wu-kan/wu-kan.github.io/blob/a94869ef1f1f6bf5daf9535cacbfc69912c2322b/_posts/2022-01-27-%E6%A8%A1%E6%8B%9F%E5%99%A8%20GPGPU-Sim%20%E7%9A%84%E4%BD%BF%E7%94%A8%E4%BB%8B%E7%BB%8D.md)
  
  - [use GPGPU-SIM](https://blog.csdn.net/gtyinstinct/article/details/126075885)

### V.2 Implement GEMM

GEMM is general Matrix Multiply.

You will implement GEMM using [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) (CUDA is a general-purpose parallel computing platform and programming model proposed by NVIDIA). And you need to check the output is right in your way.

`Hint` You can simulate your program in GPGPU-SIM, though you did't have NVIDIA GPUs. The input size is up to you.

### V.3 Submit

- Your solutions to implement GEMM

- The output of simulating your program in GPGPU-SIM
