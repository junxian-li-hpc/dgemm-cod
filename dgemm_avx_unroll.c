#include <x86intrin.h>
#include <stddef.h>

#define UNROLL 4
void dgemm_avx_unroll(size_t n, double *A, double *B, double *C)
{
	for (size_t i = 0; i < n; i += 4*UNROLL) {
		for (size_t j = 0; j < n; j++) {
			__m256d c[UNROLL];
			for (int x = 0; x < UNROLL; x++) {
				c[x] = _mm256_load_pd(C+i+x*4+j*n);
			}
			for (size_t k = 0; k < n; k++) {
				__m256d b = _mm256_broadcast_sd(B+k+j*n);
				for (int x = 0; x < UNROLL; x++) {
					c[x] = _mm256_add_pd(c[x],
						_mm256_mul_pd(_mm256_load_pd(A+i+k*n+x*4),
							b));
				}
			}
			for (int x = 0; x < UNROLL; x++) {
				_mm256_store_pd(C+i+x*4+j*n, c[x]);
			}
		}
	}
}
