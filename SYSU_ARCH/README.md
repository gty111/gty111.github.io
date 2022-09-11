# SYSU ARCH

> `dev`what to name exp
> 
> this is the dev version of SYSU ARCH

reference [GEM5 101](https://www.gem5.org/documentation/learning_gem5/gem5_101/)(add **changes** to fit current version of GEM5 and new ideas)

## Before the LAB

**There is some points you need to mind**

- **IMPORTANT:YOU NEED TO COMPLETE THIS LAB WITH GEM5 OF VERSION 22.0.0.2**

- GEM5 provides two ways of simulation (SE/FS). In this LAB, we only need to use **SE** mode.

- In addition to submitting your program, you need to provide introduction about how to use it at some times.

## I. Familiar with GEM5

 Freely read something about GEM5 on its [web page](https://www.gem5.org/).

### I.1 [Build GEM5](https://www.gem5.org/documentation/general_docs/building)

Recommend [using Docker](https://www.docker.com/)

#### I.1.1 Install Docker

For windows, [install wsl](https://docs.microsoft.com/zh-cn/windows/wsl/install) first by using 

```
wsl --install 
```

then [download Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/) and install.

> `wsl` is windows subsystem for linux. You can do most of the things you can do on Linux on WSL.
> 
> To integrate docker with wsl through `setting=>Resources=>WSL Integration`

For other systems, reference [this](https://docs.docker.com/desktop/).

#### I.1.2 Build GEM5 with Docker

First,obtain GEM5 image 

```
docker pull gcr.io/gem5-test/ubuntu-20.04_all-dependencies:v22-0
```

> Due to some reason, you may fail to pull that image, you can change `gcr.io` to `gcr.lank8s.cn` 

Then, launch the container

```
docker run -itd -v <gem5 directiory>:/gem5 <image>
```

> using `docker images` to get info about `<image>`
> 
> using `git clone https://github.com/gem5/gem5.git` to get gem5

Then,connect to the container

```
docker exec -it <container> /bin/bash
```

> using `docker ps` to  get info about `<container>`

> `Question 1`: What is the meaning of `-itd -v ...` in `docker run -itd -v <gem5 directiory>:/gem5 <image>` ?  What is the difference between `docker run` and `docker exec` ?

### I.2 Write an insteresting app(sieve)

Write a program that outputs one single integer at the end `the number of prime numbers <= N`(at default N = 100,000,000) . Compile your program as a static binary. Note that your program must achieve O(N) complexity.

#### I.2.1 Test sieve

You need to test your sieve program by building a `benchmark` program to validate its correctness and time complexity. 

> `Hint` How to validate its time complexity ?

### I.3  Use GEM5

- Run your sieve program in GEM5
  
  - choose an appropriate input size 
  
  > You should use something large enough that the application is interesting, but not too large that gem5 takes more than 10 minutes to execute a simulation.
  
  - change the CPU model from TimingSimpleCPU to MinorCPU. 
  
  > `Hint` : GEM5 won't compile MinorCPU by default. You need to add some modifications. GEM5 use `CPU_MODELS` as a parameter in the past. Try to execute `grep CPU_MODELS -R YOUR_GEM5_ROOT_DIR --exclude-dir=build` and see what you can find out.
  
  - Vary the CPU clock from 1GHz to 2GHz (in steps of 1GHz) with both CPU models. 
  
  - Change the memory configuration from DDR3_1600_x64 to DDR3_2133_x64 (DDR3 with a faster clock)

> `Question 2`: In each output, does `system.cpu.numCycles` times `system.clk_domain.clock` equals `simTicks` ? Why ?

### I.4 Submit

- Your config file (.py)

- The output (include config) under the **combination** (total is 8) of these configs

| CPU             | CPU_clock | DRAM          |
| --------------- | --------- | ------------- |
| TimingSimpleCPU | 1GHz      | DDR3_1600_x64 |
| MinorCPU        | 2GHz      | DDR3_2133_x64 |

- Your `sieve` program and corresponding `benchmark` program.

- Answer `Question 1 and 2`

## II Implement FSUBR

> `dev`implement more insts ? 

At this part, you will implement a missing x87 instruction (FSUBR).

`Hint` : You need to modify `src/arch/x86/isa/decoder/x87.isa` and `src/arch/x86/isa/insts/x87/arithmetic/subtraction.py` to implement FSUBR

### II.1 Tips about Implementing FSUBR

- Normally, the emulator itself is very complex, so we may feel that it would be difficult to modify or add our own things in the emulator. In fact, GEM5 is very extensible（OOP, Structured directory/files and assisting Python configs). And the easiest way to extend in GEM5 is imitation. For example, to implement the FSUBR instruction, we can mimic the implementation of the FSUB instruction.

- these [mannuals](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html) may be useful for understanding x86 ISA

- this [tutorial](https://www.gem5.org/documentation/learning_gem5/gem5_101/homework-2) introduce implementing FSUBR in detail

### II.2 Test your implementation

We are providing an implementation using FSUBR.

```cpp
 // ret = in2 - in1
 float subtract(float in1, float in2)
  {
    float ret = 0.0;
    asm ("fsubr %2, %0" : "=&t" (ret) : "%0" (in1), "u" (in2));
    return ret;
  }
```

You need to test your implementation by building a `benchmark` program and run it in GEM5 to validate your implementation is right.

> `Question 3` : In x87.isa, you may notice that some code like `Inst::FSUB1(Ed)`, what is the meaning of the content in parentheses? How do you know that?

### II.2 Submit

- Your `benchmark` program to validate FSUBR in GEM5.

- Any files that you made changes to implement FSUBR

- Answer `Question 3`  `optional`

## III Hotspot Analysis

The DAXPY loop (double precision aX + Y) is an oft used operation in programs that work with matrices and vectors. The following code implements DAXPY in C++11.

```cpp
  #include <cstdio>
  #include <random>

  int main()
  {
    const int N = 1000;
    double X[N], Y[N], alpha = 0.5;
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1, 2);
    for (int i = 0; i < N; ++i)
    {
      X[i] = dis(gen);
      Y[i] = dis(gen);
    }

    // Start of daxpy loop
    for (int i = 0; i < N; ++i)
    {
      Y[i] = alpha * X[i] + Y[i];
    }
    // End of daxpy loop

    double sum = 0;
    for (int i = 0; i < N; ++i)
    {
      sum += Y[i];
    }
    printf("%lf\n", sum);
    return 0;
  }
```

Usually while carrying out experiments for evaluating a design, one would like to look only at statistics for the portion of the code that is most important. To do so, typically programs are annotated so that the simulator, on reaching an annotated portion of the code, carries out functions like create a checkpoint, output and reset statistical variables.

You will add `m5_dump_reset_stats(<arg1>,<arg2>)` to the C++ code to reset stats just before the start of the DAXPY loop and just after it.

To use `m5_dump_reset_stats(<arg1>,<arg2>)`, you need to read the `util/m5/README.md` and correctly compile DAXPY program.

Run your modified DAXPY program in GEM5 and see what happens in the output. 

> `Question 4` : What is the difference in the output after you add `m5_dump_reset_stats(<arg1>,<arg2>)` ?

### III.1 Submit

- Your solution to correctly compile modified DAXPY program

- The output after you add `m5_dump_reset_stats(arg1,arg2)`

- Answer `Question 4`

## IV Implement NMRU replacement policy

### IV.1 NMRU

Replace the cache block randomly but not the recently used cache block.

### IV.2 Tips About Implementing NMRU

You need to modify under dir `src/mem/cache/replacement_policies`. 

As `II.1` said, you need to observe how `LRU` is implemented in GEM5. 

### IV.3 Compare NMRU and LRU in GEM5

Run your `sieve` program in GEM5 using NMRU or LRU and see what is the difference in the output. 

> `Question 5`: Can you write a program that runs faster (about 10% speedup) using NMRU/LRU instead of LRU/NMRU ?

### IV.4 Submit

- Any file that you made changes to implement NMRU

- Your solution to `Question 5` , including your designed program and the comparison between NMRU and LRU  `optional`

## V Explore GEMM and GPGPU-SIM `optional`

> This part of Lab is optional

### V.1 Build [GPGPU-SIM](http://www.gpgpu-sim.org/)

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

# FAQ

## 1. When building GEM5, it says `You're missing the gem5 style or commit message hook...`

then you can use 

```
chown -R <uid>:<gid> <gem5 directory>
```

> using `id` to get info about `<uid>` and `<gid>`

## 2. When using MinorCPU, it says `AttributeError: object 'BaseMinorCPU' has no attribute 'ArchMMU'`

That means you didn't compile MinorCPU in your GEM5. Try to figure out by reading `Hint`.

## 3. What are the scoring details of this LAB ?

At present, we didn't make a detailed grading rule. We think it will depends on how well  the students finish. However, the basic rule is "the more you do, the better you finish, the higher your score".

## 4. I found this LAB is different from GEM5 101. Which should I reference?

All contents are subject to this LAB. 

## 5. I found some bugs/problems in this LAB. Where should I issue them?

contact `guoty9[AT]mail2.sysu.edu.cn` 

> `dev` TODO issue at github
