---
title: Miscellaneous tips
toc: true
date: 2023-12-04 19:05:56
categories:
- Technology
tags:
- tips
---

> This passage is to log miscellaneous tips.

<!-- more -->
- [Internet's icon library and toolkit](https://fontawesome.com/)
- [Use HEU_KMS for Windows Activation](https://github.com/zbezj/HEU_KMS_Activator)
- [How to give a technical presentation](https://homes.cs.washington.edu/~mernst/advice/giving-talk.html)
- [Chatbot Arena: Benchmarking LLMs in the Wild](https://chat.lmsys.org/)
- display information about ELF files `readelf` https://www.man7.org/linux/man-pages/man1/readelf.1.html
- [Install Shadowsocks windows client](https://github.com/shadowsocks/shadowsocks-windows/releases/download/4.4.1.0/Shadowsocks-4.4.1.0.zip)
- Install Shadowsocks libev on Debian/Ubuntu (https://teddysun.com/358.html)
```
wget --no-check-certificate -O shadowsocks-libev-debian.sh https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks-libev-debian.sh
chmod +x shadowsocks-libev-debian.sh
./shadowsocks-libev-debian.sh 2>&1 | tee shadowsocks-libev-debian.log
```
- list symbols in binary using `nm` https://www.man7.org/linux/man-pages/man1/nm.1.html
- Search paper : https://dblp.org/  Search conf : https://dblp.org/db/conf/hpca
- mirror for huggingface [hf-mirror](https://hf-mirror.com/)
- mirror for github [ghproxy](https://mirror.ghproxy.com/) (`https://mirror.ghproxy.com/`)
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









