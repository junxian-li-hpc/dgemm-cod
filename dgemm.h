/*
 * Copyright (C) 2019, All rights reserved
 * Lian Haidong <lianhaidong@gmail.com>
 */

#ifndef __DGEMM_H
#define __DGEMM_H

void dgemm_c(size_t n, double *A, double *B, double *C);

void dgemm_avx(size_t n, double *A, double *B, double *C);

void dgemm_avx_unroll(size_t n, double *A, double *B, double *C);

void dgemm_avx_unroll_blk(size_t n, double *A, double *B, double *C);

void dgemm_avx_unroll_blk_omp(size_t n, double *A, double *B, double *C);

#endif	/* __DGEMM_H */
