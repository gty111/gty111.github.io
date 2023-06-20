---
title: Optimize GEMM step by step
toc: true
date: 2023-06-20 18:57:25
categories:
- Technology
tags:
- GEMM
- cutlass
---

一步步优化GEMM系列，每次引入一个优化概念并对比性能变化

![](img/gemm_vec.png)

<!-- more -->

> 点击每个标题链接跳转到对应github仓库

## 总体思路

首先我们构建了一个初级的GEMM kernel，它使用CUDA `mma.sync`指令来使用GPU tensor core单元，并对比了和cutlass算子的性能
 
上图展示了GEMM MMA的计算流程，蓝色部分代表1个block要计算的部分，蓝色部分下的每个小方块代表每个warp的计算部分，右侧青色部分代表每个warp的计算部分，青色部分下的每个小方块代表tensor core支持的分块大小，在调用tensor core之前，加载一个绿色方块和红色方块进入共享内存，之后每个warp独立同步地调用`mma.sync` 来计算每个分块的结果，其中 `M'`、`N'` 、`K'` 代表tensor core单元支持计算的GEMM维度。

baseline性能: 3.44% (相比cutlass)

## [使用向量化(vector)](https://github.com/gty111/GEMM_MMA/tree/vector)

vector分支主要介绍向量化load/store，

优化后性能: 4.74%

向量化在不同层级的表现

### cu level

```
*reinterpret_cast<float4*>(&A[tileIdx]) = *reinterpret_cast<float4*>(&arg.A[rowA_0*arg.problem_size.k()+colA_0]);
```

### ptx level

```
ld.global.v4.u32 {%r161, %r162, %r163, %r164}, [%rd5];
st.shared.v4.u32 [%r16], {%r161, %r162, %r163, %r164};
```

### SASS level

```
LDG.E.128 R8, [R10.64] ;
STS.128 [R17], R8 ;
```

### 结果

```
[        problem size] (8192,8192,8192)
[          cutlassMMA] Runtime: 15.816908(ms) Gflops: 69514.954246
[            MMA_base] Runtime: 460.150970(ms) Gflops: 2389.458457
[       MMA_base==ref] PASS
[             MMA_vec] Runtime: 333.618652(ms) Gflops: 3295.713894
[        MMA_vec==ref] PASS
```

## [避免bank冲突并且合并访存(bfco)](https://github.com/gty111/GEMM_MMA/tree/bfco)

bfco分支主要介绍如何通过解决shared memory bank conflict 和 memory coalesce (访存合并) 来优化性能

优化后性能: 5.00%

### Shared memory bank

[参考cuda programming guide](https://docs.nvidia.cn/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x)

![](img/smem_bank.png)

要注意连续的bank存储连续的字(32-bits)，这里字的大小为32 bits，总共有32个bank

要想解决bank conflict问题，要将一个warp内线程读取的shared memory尽量分散到不同的bank里

### memory coalesce（访存合并）

访存合并用一句话来简单概括就是一个warp内线程读取的global memory尽量是连续的且128字节对齐

为什么是128字节对齐而不是其他数字？我的理解是cache line的大小是128字节，这样一个warp内的访存可以合并成以cache line为基本单位的memory transaction

### 代码分析

为了解决bank conflict 和 memory coalesce，对代码做的主要修改为变量 `tileidx`

```cpp
// in function loadtileC
int tileIdx = threadIdx.x*64 + i*4; // base
int tileIdx = threadIdx.x*64 + (i+threadIdx.x/1)%16*4; // bank free
int tileIdx = threadIdx.x*4 + i*blockDim.x*4; // memory coalesce + bank free
```

### 结果

```
[        problem size] (8192,8192,8192)
[          cutlassMMA] Runtime: 15.788442(ms) Gflops: 69640.288231
[            MMA_base] Runtime: 333.625763(ms) Gflops: 3295.643652
[       MMA_base==ref] PASS
[              MMA_bf] Runtime: 326.514526(ms) Gflops: 3367.420249
[         MMA_bf==ref] PASS
[           MMA_bf_co] Runtime: 315.669495(ms) Gflops: 3483.110172
[      MMA_bf_co==ref] PASS
```

## [使用异步拷贝(ldgsts)](https://github.com/gty111/GEMM_MMA/tree/ldgsts)

ldgsts 分支主要来介绍使用Ampere引入的异步拷贝来优化性能

优化后性能: 5.36%

### 异步拷贝

CUDA 11 includes a new asynchronous copy (async copy) API to take advantage of the A100 
GPU’s hardware-accelerated direct-copy-to-shared functionality. Async copy performs an 
asynchronous (non-blocking) direct memory transfer from global memory to shared memory, 
bypassing the SM threads and combining the functions of separate “load from global memory 
into a register”, and “write to shared memory from a register” operations into a single, efficient 
operation. 

Async copy eliminates the need for intermediate staging of data through the register file (RF), 
reducing register file bandwidth. It also efficiently uses memory bandwidth and reduces power 
consumption. As the name implies, async copy works asynchronously, allowing other 
computations to occur during global-to-shared memory copies. Async copy is able to notify the 
program of copy completion via the GPU’s new barrier feature.

Bypassing L1 and the register file can significantly accelerate memory copy performance, 
especially for multiple successive async-copy operations that copy large amounts of data from 
global to shared memory. 

![](img/async_copy.png)

Two variants of the async copy instruction are available for different usage scenarios. BYPASS, 
which bypasses L1 cache and the register file as described above, and ACCESS which saves 
data to L1 for subsequent accesses and reuse. 

### cu level

```
asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(&C[tileIdx])),
                "l"(&arg.C[rowC_0*arg.problem_size.n()+colC_0]),
                "n"(16)
            );
```

### ptx level

```
cp.async.cg.shared.global [%r158], [%rd18], 16;
```

### SASS level

```
LDGSTS.E.BYPASS.128 [R7+0x1800], [R4.64] ;
```


### 结果

```
[        problem size] (8192,8192,8192)
[          cutlassMMA] Runtime: 15.938765(ms) Gflops: 68983.491336
[            MMA_base] Runtime: 315.683228(ms) Gflops: 3482.958649
[       MMA_base==ref] PASS
[            MMA_ldgsts] Runtime: 297.315948(ms) Gflops: 3698.125289
[       MMA_ldgsts==ref] PASS
```

## [使用寄存器(reg)](https://github.com/gty111/GEMM_MMA/tree/reg)

reg 分支介绍使用寄存器来优化性能

优化后性能: 35.39%

### CUDA中的寄存器

寄存器的概念可能对于高级编程者来说是比较陌生的，因为在编程中一般并不会刻意地声明要使用寄存器来做什么操作，编译器会帮我们处理好这个问题，这就导致了在编写CUDA算子时往往会忽略掉寄存器的使用，可以通过ncu或编译时设置编译参数来查看kernel中每个线程使用了几个寄存器，比如在我们对比的cutlass的kernel中每个线程使用了230个寄存器，但是本例中baseline的kernel中每个线程只使用了32个寄存器，所以可以考虑将频繁使用的`tileC`(也就是图中的蓝色部分)从共享内存转移到寄存器中。

如何使用寄存器？其实很简单，在kernel中声明变量或数组就可以(不过如果一个线程使用太多寄存器会发生register spilling，可以在编译好程序后反汇编查看下有没有local memory)

在代码中添加了
```
ElementOutput C_fragment[64];
```
并修改好相关逻辑后，再次编译发现每个线程使用了156个线程

### 杂项
之前方法优化效果不明显的原因应该是kernel性能瓶颈在别的地方，可以从这个寄存器优化后的版本尝试如果不采用向量化，不解决bank冲突或访存不合并，或者不采用异步拷贝，kernel的性能变化是怎样的？

### 结果

原来没有使用寄存器才是kernel性能差的主要原因...

```
[        problem size] (8192,8192,8192)
[          cutlassMMA] Runtime: 16.149094(ms) Gflops: 68085.036418
[            MMA_base] Runtime: 297.333862(ms) Gflops: 3697.902483
[       MMA_base==ref] PASS
[            MMA_tune] Runtime: 45.636402(ms) Gflops: 24092.863952
[       MMA_tune==ref] PASS
```

## [使用数据预取(prefetch)](https://github.com/gty111/GEMM_MMA/tree/prefetch)

prefetch 分支介绍使用数据预取来优化性能

优化后性能：39.36%

数据预取需要将缓冲区加倍，主要流程如下

假设计算`mma_{i}`依赖于数据`data_{i}`, `load data_{i}`代表开始加载数据`data_{i}`, 只有在`synchronize`后加载的数据才保证可见, 那么数据预取的伪代码如下

```
load data_{1}

for i=1:...
    synchronize
    mma_{i}
    load data_{i+1}
end
```
这样可以让数据加载(data_{i+1})和计算(mma_{i})尽可能重叠起来

### 结果

```
[        problem size] (8192,8192,8192)
[          cutlassMMA] Runtime: 15.947670(ms) Gflops: 68944.969952
[            MMA_base] Runtime: 45.381512(ms) Gflops: 24228.184273
[       MMA_base==ref] PASS
[            MMA_tune] Runtime: 40.519497(ms) Gflops: 27135.372140
[       MMA_tune==ref] PASS
```


## [关于PTXAS有趣的发现(ptxas)](https://github.com/gty111/GEMM_MMA/tree/ptxas)

ptxas 分支分享一个调优过程中发现的关于ptxas(ptx汇编器)有意思的东西

### 事情起因

在优化kernel过程中，发现每个warp循环计算每个Ctile时，都需要从共享内存加载Atile和Btile一次，这样Atile和Btile会被重复加载4次，那其实这里数据是可以复用的，而不用多次加载，于是重新设计了下每个warp计算流程，将4x4tile划分为4个2x2的大tile，每次计算大tile前先把对应的Atile和Btile从共享内存加载到寄存器中，在计算下一个大tile时只需要重新加载一个 A/B tile即可(例如从左上角移动到右上角只需要重新加载Btile即可，Atile是可以复用的)，下图为优化前后的计算流程图，其中C tile为每个warp需要计算的矩阵C的部分，图上的数字代表数据块被加载的次数

![](img/mma_dataflow.png)

然而就在我把代码写好，验证通过后，惊讶地发现两个kernel的运行时间很接近，通过ncu profile后发现运行的指令数竟然是一样的，反汇编后发现两个kernel的sass指令数竟然是相同的，然后仔细看了下，代码逻辑完全是一模一样的，只是部分寄存器命名不一样，这有点离谱，然后看了下两个kernel的ptx指令逻辑还是不一样的，难道CUDA的ptxas优化的这么离谱。

这里给出kernel的ptx和sass指令

两个kernel的ptx版本分别为 [ptx_mma](https://github.com/gty111/GEMM_MMA/blob/ptxas/ptx/ptx_mma) 和 [ptx_mma_tune](https://github.com/gty111/GEMM_MMA/blob/ptxas/ptx/ptx_mma_tune)

两个kernel的sass版本分别为 [sass_mma](https://github.com/gty111/GEMM_MMA/blob/ptxas/sass/sass_mma) 和 [sass_mma_tune](https://github.com/gty111/GEMM_MMA/blob/ptxas/sass/sass_mma_tune)

### 杂项

优化了半天最后发现机器码都是一样的，确实感觉到了编译器的强大，关键是怎么知道代码哪些是已经被编译器优化好了呢。

另外意外发现了A100 80GB 相比A100 40GB 可以提升33%左右的性能，于是感觉很奇怪，这两个版本不就是显存大小不一样嘛，怎么运行速度差距这么大，于是发现A100 80GB显存带宽 2TB/s，而40BG版本显存带宽 1.5TB/s，这相当于显存带宽提升了33%，这难道全部转化成性能提升了吗？

```
A100 40GB
[        problem size] (5120,4096,4096)
[          cutlassMMA] Runtime: 2.370118(ms) Gflops: 72485.278929
[            MMA_base] Runtime: 6.451385(ms) Gflops: 26629.735875
[       MMA_base==ref] PASS
[            MMA_tune] Runtime: 6.456460(ms) Gflops: 26608.804078
[       MMA_tune==ref] PASS

A100 80GB
[        problem size] (5120,4096,4096)
[          cutlassMMA] Runtime: 1.781453(ms) Gflops: 96437.410102
[            MMA_base] Runtime: 4.881101(ms) Gflops: 35196.711561
[       MMA_base==ref] PASS
[            MMA_tune] Runtime: 4.883047(ms) Gflops: 35182.685107
[       MMA_tune==ref] PASS
```