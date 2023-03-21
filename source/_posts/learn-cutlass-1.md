---
title: learn-cutlass-1
date: 2023-03-21 09:00:19
categories:
- Technology
tags:
- cutlass
---

In cutlass 3.0, it introduces a new library, Cute, to describe and manipulate tensors of threads and data. I think the core of cutlass is GEMM(or other computations) and data movement.

## RowMajorInterleaved (ColumnMajorInterleaved)

```c++
#include "cutlass/layout/matrix.h"
template<int Interleave> struct cutlass::layout::RowMajorInterleaved<Interleave>;
```

RowMajorInterleaved is a layout which confused me. I didn't know the meaning of Interleaved.So I create an example to figure it out.

<!-- more -->

```c++

#include <iostream>
#include <cstdio>

// Defines cutlass::layout::RowMajorInterleave
#include "cutlass/layout/matrix.h"

// Defines cutlass::HostTensor<>
#include "cutlass/util/host_tensor.h"

// Defines cutlass::MatrixCoord
#include "cutlass/matrix_coord.h"

#define M 4
#define N 4

int main(){
    cutlass::HostTensor<int,cutlass::layout::RowMajorInterleaved<2> > A(cutlass::MatrixCoord(M,N));
    
    int num = 0;
    for(int i=0;i<M;i++)
    for(int j=0;j<N;j++){
        A.at({i,j}) = ++num; 
    }

    int *A_ = A.host_data();
    for(int i=0;i<A.capacity();i++){
        printf("%3d ",A_[i]);
        // if((i+1)%N==0)printf("\n");
    }
    /**
     *  output:
     *  1 5 2 6 3 7 4 8 9 13 10 14 11 15 12 16
     *  
    */
}
```

If tensor A is a simple RowMajor, the output should be this

```
 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
```
In my opinion, `Interleaved` means it will iterate in shape(1) with size `Interleave` and then iterate in shape(0). 
Other things need to mind is `Interleaved` may cause padding of a matrix, like

```c++
#define M 3
#define N 3
cutlass::HostTensor<int,cutlass::layout::RowMajorInterleaved<2> > A(cutlass::MatrixCoord(M,N));
int num = 0;
for(int i=0;i<M;i++)
for(int j=0;j<N;j++){
    A.at({i,j}) = ++num; 
}
/**
 * the element in A should be 
 * 1 4 2 5 3 6 7 0 8 0 9 0
```

## 05_batched_gemm

Batched gemm can be illustrated as follows
![](/img/batched_gemm.jpg)

