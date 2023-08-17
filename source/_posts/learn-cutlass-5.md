---
title: learn-cutlass-5
toc: true
date: 2023-05-14 11:48:32
categories:
- Technology
tags:
- cutlass
---

Cutlass use abstract `layout` to express the mapping rules from logic index to physical index. 

<!-- more -->

## Affine2 
> 18_amphere_fp64_tensorop_affine2_gemm

Affine2 is a speical layout in cutlass.

In the normal GEMM, the fast changing dimension of a matrix always has stride 
equals to 1, e.g. ColumnMajor and RowMajor matrix.  Affine2 matrix can have 
larger than 1 stride in both dimensions.  To support such layout, we need to 
change to method to visit the global memory:

  1. We can only visit 1 element a time because elements are not stored
     consecutively anymore.  Vectorized load/store is not possible.
  2. One extra multiplication is needed in calculating the global memory
     address
     addr = base_pointer + coord1 * stride1 + coord2 * stride2

The explanation is a little abstract, let's create an example to illustrate it.

```cpp

#include <iostream>

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"

using ElementInputA = double;                      // Data type of elements in input tensor

using LayoutInputA = cutlass::layout::AffineRank2ColumnMajor;

int main() {

  // Construct Gemm ProblemSize with user defined output size
  cutlass::gemm::GemmCoord problem_size = {4, 4, 4};

  typename LayoutInputA::Stride::Index stride_factor_A[] = {2, 2}; 

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk(),
         cutlass::layout::Affine2Layout_Factory<LayoutInputA>::layout_factory(problem_size.mk(),
                                                                              stride_factor_A));

  // Fill input and output matrices on host using CUTLASS helper functions
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(100),
      ElementInputA(-100),
      0);  // <- Fill matrix A on host with uniform-distribution random data

  std::cout << tensor_a.host_view() << "\n\n";
  std::cout << tensor_a.capacity() << "\n";
  ElementInputA *a = tensor_a.host_data();
  for(int i=0;i<tensor_a.capacity();i++){
    std::cout << a[i] << ' ';
  }
  std::cout << '\n';
}

```

And the output should be

```
68, -21, 56, 59,
82, -60, -32, 53,
-44, 10, -4, 25,
-27, 2, 90, 83

64
68 0 82 0 -44 0 -27 0 0 0 0 0 0 0 0 0 -21 0 -60 0 10 0 2 0 0 0 0 0 0 0 0 0 56 0 -32 0 -4 0 90 0 0 0 0 0 0 0 0 0 59 0 53 0 25 0 83 0 0 0 0 0 0 0 0 0
```

So affine2 is a layout that builds a submatrix through extracting original matrix based on the given stride.

## Quaternion
> 21_quaternion_gemm

Quaternion is an interesting concept mostly used in computer graphics. In my opinion, it can be seen as analogy to complex number.
[The detailed information about quaternion can be found here.](https://github.com/Krasjet/quaternion/blob/master/quaternion.pdf)