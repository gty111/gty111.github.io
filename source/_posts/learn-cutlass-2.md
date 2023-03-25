---
title: learn-cutlass-2
toc: true
date: 2023-03-24 14:29:03
categories:
- Technology
tags:
- cutlass
---

I always wonder why cutlass provides many kinds of implementions of GEMM instead of just only one. In my opinion, in different situations the best implementions of GEMM differs. So that is what differs cutlass from cublas. You can make your own custiomlized implemention of GEMM to provide the best performance.

<!-- more -->

## InstructionShape
I think it is only used in tensor core operations for specifying the basic GEMM (M,N,K) such as 
```c++ 
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>; // TensorCore instruction shape
```
You can view [this](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes) for more info.

