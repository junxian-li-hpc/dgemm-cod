#include <x86intrin.h>
#include <stddef.h>

#define UNROLL 4
#define BLOCKSIZE 32

static inline void do_block(int n, int si, int sj, int sk,
                            double *A, double *B, double *C)
{
  for (int i = si; i < si + BLOCKSIZE; i += UNROLL * 4)
  {
    for (int j = sj; j < sj + BLOCKSIZE; j++)
    {
      __m256d c[UNROLL];

      for (int x = 0; x < UNROLL; x++)
      {
        c[x] = _mm256_load_pd(C + i + x * 4 + j * n);
      }
      for (int k = sk; k < sk + BLOCKSIZE; k++)
      {
        __m256d b = _mm256_broadcast_sd(B + k + j * n);
        for (int x = 0; x < UNROLL; x++)
        {
          c[x] = _mm256_add_pd(c[x],
                               _mm256_mul_pd(
                                   _mm256_load_pd(A + n * k + x * 4 + i), b));
        }
      }

      for (int x = 0; x < UNROLL; x++)
      {
        _mm256_store_pd(C + i + x * 4 + j * n, c[x]);
      }
    }
  }
}

void dgemm_avx_unroll_blk(size_t n, double *A, double *B, double *C)
{
  // #pragma omp parallel for
  for (int sj = 0; sj < n; sj += BLOCKSIZE)
  {
    for (int si = 0; si < n; si += BLOCKSIZE)
    {
      for (int sk = 0; sk < n; sk += BLOCKSIZE)
      {
        do_block(n, si, sj, sk, A, B, C);
      }
    }
  }
}
