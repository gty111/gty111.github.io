---
layout: default
title: IV.Implement NMRU replacement policy
nav_order: 6
---

## IV Implement NMRU replacement policy

### IV.1 NMRU

Replace the cache block randomly but not the recently used cache block.

### IV.2 Tips About Implementing NMRU

You need to modify under dir `src/mem/cache/replacement_policies`.

As `II.1` said, you need to observe how `LRU` is implemented in GEM5.

### IV.3 Compare NMRU and LRU in GEM5

Run your `sieve` program in GEM5 using NMRU or LRU and see what is the difference in the output.

{: .highlight}
> `Question 5`: Can you write a program that runs faster (about 10% speedup) using NMRU/LRU instead of LRU/NMRU ?

### IV.4 Submit

- Any file that you made changes to implement NMRU

- Your solution to `Question 5` , including your designed program and the comparison between NMRU and LRU `optional`
