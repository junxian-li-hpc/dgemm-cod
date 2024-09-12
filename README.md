# dgemm

本文以N*N 矩阵乘法为例叙述优化的过程，该代码运行于x86 Linux/Mac 平台。 整个优化分为一下几个步骤：
- 向量化 （在x86架构上采用AVX）
- 循环展开（loop unrolling）
- cache blocking （也可称为tiling）
- 多线程 （OpenMP）

NOTE： 
- 本文的代码来自 << Computer Organization and Design RISC-V edition>>.

