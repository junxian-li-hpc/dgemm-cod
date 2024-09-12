/*
 * Copyright (C) 2019, All rights reserved
 * Lian Haidong <lianhaidong@gmail.com>
 */

#include <x86intrin.h>
#include <stddef.h>

void dgemm_avx(size_t n, double *A, double *B, double *C)
{
	for (size_t i = 0; i < n; i += 4) {
		for (size_t j = 0; j < n; j++) {
			__m256d c0 = _mm256_load_pd(C+i+j*n); /* c0 = C[i][j] */
			for (size_t k = 0; k < n; k++) {
				c0 = _mm256_add_pd(c0,
					_mm256_mul_pd(_mm256_load_pd(A+i+k*n),
						_mm256_broadcast_sd(B+k+j*n)));
			}
			_mm256_store_pd(C+i+j*n, c0);  /* C[i][j] = c0 */;
		}
	}
}
