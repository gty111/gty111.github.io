---
title: learn-cutlass-4
toc: true
date: 2023-04-16 11:05:57
categories:
- Technology
tags:
- cutlass
---

![](img/13_example_fusion.png)
[Cutlass examples](https://github.com/NVIDIA/cutlass/tree/main/examples) gives us many examples to learn cutlass. At this time, 13_two_tensor_op_fusion is introduced.

<!-- more -->

## What is "two tensor op fusion"? 

Fusing two back to back GEMMs/Convolutions into one kernel.

## Why fusing? 

To avoid intermediate results round trip to memory.

## How fusing? 

Cutlass uses threadblockTile and WarpTile to respresent the shape of final result matrix which is calculated by a threadblock or warp respectively. 

- First, if we want to fuse two kernel of GEMM , the number of threadblocks of two kernels must be the same. 
- Second, to use the intermediate result the dimension M of two GEMM must be the same. `Otherwisely, the input of second GEMM's threadblock may lie at other threadblock.` Similarly, the dimension N of GEMM and the dimension N of threadblock must be the same. 
![](img/13_example_block_resident_fusion.png) 
thread_block_tile_N = problem_N
- Third, the dimension N of threadblock and the dimension N of warp must be the same when using register, `which can be relaxed when using share memory because different warps in a threadblock can exchange data with each other`.
![](img/13_example_rf_resident_fusion.png) 
warp_tile_N = thread_block_tile_N
![](img/13_example_shmem_resident_fusion.png) 
relaxed constraints
