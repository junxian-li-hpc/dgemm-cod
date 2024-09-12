cmake_minimum_required(VERSION 3.10)


set(SRCS dgemm.c
         dgemm_c.c
         dgemm_avx.c
         dgemm_avx_unroll.c
         dgemm_avx_unroll_blk.c
         dgemm_avx_unroll_blk_omp.c
)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -fopenmp")
add_executable(dgemm_test ${SRCS})


