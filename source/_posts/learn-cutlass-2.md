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

---

The annotation in cutlass:

When the template variables are passed to instantiate CUTLASS GEMM kernel, it internally deduce the amount of threads needed per thread-block, amount of shared memory, storing data in bank-conflict free manner, and ton of other variables required to compose, initialize and launch a high performance GEMM kernel. This is the beauty of CUTLASS, it relieves developer from understanding and coding complicated hardware optimizations which can easily go wrong.

CUTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp and thread-block level, they compute on their own tile-size with higher level of tile sizes being composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute threadblock-tile (tile size computed by a threadblock).

## InstructionShape
When it is used in tensor core operations for specifying the basic GEMM (M,N,K) such as 
```c++ 
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>; // TensorCore instruction shape
```
Or it is used in SIMT operations 
```c++
// SIMT (except dp4a)
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
// SIMT dp4a
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;
```

## dp4a

```
dp4a.atype.btype d, a, b, c;
.atype = .btype = { .u32, .s32 };
```

Four-way byte dot product which is accumulated in 32-bit result.
Operand a and b are 32-bit inputs which hold 4 byte inputs in packed form for dot product.
Operand c has type .u32 if both .atype and .btype are .u32 else operand c has type .s32.

```
d = c;
∕∕ Extract 4 bytes from a 32bit input and sign or zero extend
∕∕ based on input type.
Va = extractAndSignOrZeroExt_4(a, .atype);
Vb = extractAndSignOrZeroExt_4(b, .btype);
for (i = 0; i < 4; ++i) {
d += Va[i] * Vb[i];
}
```

## [Epilogue](https://github.com/NVIDIA/cutlass/blob/master/media/docs/efficient_gemm.md#epilogue)

The above code focuses only on the matrix multiply computation C = AB whose result is held in the registers of each thread within the threadblock. The mapping of logical elements in the output tile to each thread is chosen to maximize performance of the matrix multiply computation but does not result in efficient, coalesced loads and stores to global memory.

The epilogue is a separate phase in which threads exchange data through shared memory then cooperatively access global memory using efficient striped access patterns. It is also the phase in which linear scaling and other elementwise operations may be conveniently computed using the matrix product results as inputs.

CUTLASS defines several typical epilogue operations such as linear scaling and clamping, but other device-side function call operators may be used to perform custom operations.

## 06_splitK_gemm

`splitK` is partitioning a GEMM with its K dimension.

![](/img/splitK.svg)

\\(C = \sum_{i=1}^{K}{A_i}*B_i\\)

```c++
// Define cutlass::gemm::device::GemmSplitKParallel
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
```

In cutlass, templates are arguments/context of a function. Not all the combinations of templates work. So you need to know which combination is correct. To know its more, try to build a splitK using arch sm80 instead of sm70. One possible solution is as follows:
```
142c142
< using SmArch = cutlass::arch::Sm70;
---
> using SmArch = cutlass::arch::Sm80;
150c150
< using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4
---
> using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 16
188,189c188,189
<   if (props.major != 7) {
<     std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
---
>   if (props.major != 8 && props.minor != 0) {
>     std::cerr << "Amphere Tensor Ops must be run on a machine with compute capability of 80."
```

