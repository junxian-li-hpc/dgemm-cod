
all: dgemm
dgemm:
	gcc -O3 -mavx  -mfma -o dgemm dgemm.c dgemm_c.c dgemm_avx.c dgemm_avx_unroll.c dgemm_avx_unroll_blk.c  dgemm_avx_unroll_blk_omp.c

run:
	./dgemm