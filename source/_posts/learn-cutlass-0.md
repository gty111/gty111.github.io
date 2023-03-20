---
title: learn-cutlass-0
date: 2023-03-20 20:44:12
categories: 
- Technology
tags:
- cutlass
---

> learn cutlass is a series of tutorials to learn cutlass by reading its examples or source code

CUTLASS is a header-only template library. After reading that, you will **be lost in templates**.

## 00_basic_gemm

```c++
// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

using CutlassGemm = cutlass::gemm::device::Gemm<A_TYPE,A_LAYOUT,B_TYPE,B_LAYOUT,C_TYPE,C_LAYOUT> ;
// where A_TYPE is Data-type of A matrix and A_LAYOUT is Layout of A matrix

CutlassGemm gemm_operator;

CutlassGemm::Arguments args({M , N, K},  
                            {A_POINTER, lda},    
                            {B_POINTER, ldb},   
                            {C_POINTER, ldc},    
                            {C_POINTER, ldc},    
                            {alpha, beta});
// where A_POINTER is pointer of A matrix and lda is the number of elements between consecutive rows or colmns

cutlass::Status status = gemm_operator(args);
// call gemm operation

```

## 01_cutlass_utilities

```c++
// CUTLASS includes needed for half-precision GEMM kernel
#include "cutlass/cutlass.h"
#include "cutlass/core_io.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"

//
// CUTLASS utility includes
//

// Defines operator<<() to write TensorView objects to std::ostream
#include "cutlass/util/tensor_view_io.h"

// Defines cutlass::HostTensor<>
#include "cutlass/util/host_tensor.h"

// Defines cutlass::half_t
#include "cutlass/numeric_types.h"

// Defines device_memory::copy_device_to_device()
#include "cutlass/util/device_memory.h"

// Defines cutlass::reference::device::TensorFillRandomGaussian()
#include "cutlass/util/reference/device/tensor_fill.h"

// Defines cutlass::reference::host::TensorEquals()
#include "cutlass/util/reference/host/tensor_compare.h"

// Defines cutlass::reference::host::Gemm()
#include "cutlass/util/reference/host/gemm.h"


// another way to call gemm without using Arguments
cutlass::Status status = gemm_op({
    {M, N, K},
    {A, lda},
    {B, ldb},
    {C, ldc},
    {C, ldc},
    {alpha, beta}
  });

// define a tensor (M,N) in cutlass, where DTYPE is data type
cutlass::HostTensor<DTYPE,LAYOUT> VAR(cutlass::MatrixCoord(M,N)) ;
cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> A(cutlass::MatrixCoord(M, K));

// fill a tensor (RandomGaussian) where A.device_view() return TensorView of that tensor in cutlass
cutlass::reference::device::TensorFillRandomGaussian(
    A.device_view(),
    seed,
    mean,
    stddev,
    bits_less_than_one
  );

// copy data from device to device in cutlass where A.device_data() return pointer of that tensor
// A.capacity() return the logical capacity based on extent and layout. May differ from size().
cutlass::device_memory::copy_device_to_device(
    C_reference.device_data(), 
    C_cutlass.device_data(), 
    C_cutlass.capacity());

// Copies data from device to host
A.sync_host();

// Copies data from host to device
A.sync_device();

// Compute the reference result using the host-side GEMM reference implementation.
// I think the only difference between TensorView and TensorRef is that TensorView is read-only 
// while TensorRef can return pointer of matrix
cutlass::reference::host::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::ColumnMajor,              // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementOutput
    cutlass::layout::ColumnMajor,              // LayoutOutput
    cutlass::half_t,                           // ScalarType
    cutlass::half_t                            // ComputeType
> gemm_ref;

gemm_ref(
    {M, N, K},                          // problem size (type: cutlass::gemm::GemmCoord)
    alpha,                              // alpha        (type: cutlass::half_t)
    A.host_ref(),                       // A            (type: TensorRef<half_t, ColumnMajor>)
    B.host_ref(),                       // B            (type: TensorRef<half_t, ColumnMajor>)
    beta,                               // beta         (type: cutlass::half_t)
    C_reference.host_ref()              // C            (type: TensorRef<half_t, ColumnMajor>)
);

// Compare reference to computed results
cutlass::reference::host::TensorEquals(
    C_reference.host_view(), 
    C_cutlass.host_view());
```

## 04_tile_iterator

```c++
// CUTLASS includes
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

//
//  CUTLASS utility includes
//

// Defines operator<<() to write TensorView objects to std::ostream
#include "cutlass/util/tensor_view_io.h"

// Defines cutlass::HostTensor<>
#include "cutlass/util/host_tensor.h"

// Defines cutlass::reference::host::TensorFill() and
// cutlass::reference::host::TensorFillBlockSequential()
#include "cutlass/util/reference/host/tensor_fill.h"


// For this example, we chose a <64, 4> tile shape. The PredicateTileIterator expects
// PitchLinearShape and PitchLinear layout.
using Shape = cutlass::layout::PitchLinearShape<64, 4>;
using Layout = cutlass::layout::PitchLinear;
using Element = int;
int const kThreads = 32;

// ThreadMaps define how threads are mapped to a given tile. The PitchLinearStripminedThreadMap
// stripmines a pitch-linear tile among a given number of threads, first along the contiguous
// dimension then along the strided dimension.
using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;

// Define the PredicateTileIterator, using TileShape, Element, Layout, and ThreadMap types
using Iterator = cutlass::transform::threadblock::PredicatedTileIterator<
    Shape, Element, Layout, 1, ThreadMap>;


cutlass::Coord<2> copy_extent = cutlass::make_Coord(M, K);
cutlass::Coord<2> alloc_extent = cutlass::make_Coord(M, K);

// another way to define tensor
// Allocate source and destination tensors
cutlass::HostTensor<Element, Layout> src_tensor(alloc_extent);
cutlass::HostTensor<Element, Layout> dst_tensor(alloc_extent);

Element oob_value = Element(-1);

// Initialize destination tensor with all -1s
cutlass::reference::host::TensorFill(dst_tensor.host_view(), oob_value);
// Initialize source tensor with sequentially increasing values
cutlass::reference::host::BlockFillSequential(src_tensor.host_data(), src_tensor.capacity());

dst_tensor.sync_device();
src_tensor.sync_device();

typename Iterator::Params dst_params(dst_tensor.layout());
typename Iterator::Params src_params(src_tensor.layout());

dim3 block(kThreads, 1);
dim3 grid(1, 1);

// Launch copy kernel to perform the copy
copy<Iterator><<< grid, block >>>(
        dst_params,
        dst_tensor.device_data(),
        src_params,
        src_tensor.device_data(),
        copy_extent
);

// copy function
template <typename Iterator>
__global__ void copy(
    typename Iterator::Params dst_params,
    typename Iterator::Element *dst_pointer,
    typename Iterator::Params src_params,
    typename Iterator::Element *src_pointer,
    cutlass::Coord<2> extent) {

    Iterator dst_iterator(dst_params, dst_pointer, extent, threadIdx.x);
    Iterator src_iterator(src_params, src_pointer, extent, threadIdx.x);

    // PredicatedTileIterator uses PitchLinear layout and therefore takes in a PitchLinearShape.
    // The contiguous dimension can be accessed via Iterator::Shape::kContiguous and the strided
    // dimension can be accessed via Iterator::Shape::kStrided
    int iterations = (extent[1] + Iterator::Shape::kStrided - 1) / Iterator::Shape::kStrided;

    typename Iterator::Fragment fragment;

    for(; iterations > 0; --iterations) {
      src_iterator.load(fragment);
      dst_iterator.store(fragment);

      ++src_iterator;
      ++dst_iterator;
    }
}
```





