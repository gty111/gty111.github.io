---
title: Tips for Linux and Programming
toc: true
date: 2023-03-30 19:05:56
categories:
- Technology
tags:
- linux
---

> This passage is to log some tips about Linux.

<!-- more -->
- mirror for github [ghproxy](https://ghproxy.com/)
- [Comparison between source and assembly instruction](https://godbolt.org/)
- [C alternative tokens](https://en.wikipedia.org/wiki/C_alternative_tokens) : use `and` to replace `&&` etc.
- compilation flags
    - -E : Only run the preprocessor
    - -S : Only run preprocess and compilation steps (=> .s .ll)
    - -c : Only run preprocess, compile, and assemble steps (=> .o)
- [-nostdlib](https://gcc.gnu.org/onlinedocs/gcc/Link-Options.html#index-nostdlib)
- view system INFO : `cat /proc/version` (kernel) or `cat /etc/issue` (system)
- view cpu INFO : `cat /proc/cpuinfo`
- make -n : only print inst not execute 
- compress : `tar -cf *.tar path-to-file(dir)` uncompress : `tar -xf *.tar`
- [ls](https://www.runoob.com/linux/linux-comm-ls.html)
- [/etc/passwd](https://www.geeksforgeeks.org/understanding-the-etc-passwd-file/)
- [How to use ssh](https://zhuanlan.zhihu.com/p/21999778)
- kown_hosts : log the public key of the host you have visited
- How to change default shell : `chsh`









