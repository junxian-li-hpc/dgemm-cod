#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>

#define BLOCKSIZE 32
#define UNROLL 4

double time_ms();

void dgemm(int n, double *A, double *B, double *C)
{
  memset(C, 0, sizeof(double) * n * n); // 初始化 C 矩阵为 0

  // 处理所有的 BLOCKSIZE x BLOCKSIZE 子块
  for (int i = 0; i < n; i += BLOCKSIZE)
  {
    for (int j = 0; j < n; j += BLOCKSIZE)
    {
      for (int k = 0; k < n; k += BLOCKSIZE)
      {
        // 计算子块 C[i:j][j:k]
        int i_max = i + BLOCKSIZE > n ? n : i + BLOCKSIZE;
        int j_max = j + BLOCKSIZE > n ? n : j + BLOCKSIZE;
        int k_max = k + BLOCKSIZE > n ? n : k + BLOCKSIZE;

        for (int mi = i; mi < i_max; mi++)
        {
          for (int ki = k; ki < k_max; ki++)
          {
            double aval = A[mi * n + ki];
            __m256d aavx2 = _mm256_set1_pd(aval);

            // 处理连续 4 倍数的部分，每次处理 4*UNROLL = 16 个元素
            int jj = j;
            for (; jj <= j_max - UNROLL * 4; jj += UNROLL * 4)
            {
              for (int u = 0; u < UNROLL * 4; u += 4)
              {
                __m256d bavx = _mm256_loadu_pd(&B[ki * n + jj + u]);
                __m256d cavx = _mm256_loadu_pd(&C[mi * n + jj + u]);

                cavx = _mm256_fmadd_pd(aavx2, bavx, cavx); // C += A * B
                _mm256_storeu_pd(&C[mi * n + jj + u], cavx);
              }
            }

            // 处理剩余部分
            for (; jj < j_max; jj++)
            {
              C[mi * n + jj] += aval * B[ki * n + jj];
            }
          }
        }
      }
    }
  }
}

int main(int argc, char **argv)
{
  int n = (argc > 1) ? atoi(argv[1]) : 1000;
  if (n <= 0)
  {
    fprintf(stderr, "Invalid matrix size. Please enter a positive integer.\n");
    return 1;
  }

  double *A = (double *)malloc(n * n * sizeof(double));
  double *B = (double *)malloc(n * n * sizeof(double));
  double *C = (double *)malloc(n * n * sizeof(double));
  double *C_golden = (double *)malloc(n * n * sizeof(double));

  if (!A || !B || !C)
  {
    fprintf(stderr, "Memory allocation failed.\n");
    free(A);
    free(B);
    free(C);
    return 1;
  }

  // Initialize matrices A and B
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
    {
      A[i * n + j] = i + j;
      B[i * n + j] = i + j;
    }

  memset(C_golden, 0, sizeof(double) * n * n);
#pragma omp parallel for
  for (int mi = 0; mi < n; mi++)
  {
    for (int ni = 0; ni < n; ni++)
    {
      for (int ki = 0; ki < n; ki++)
        C_golden[mi * n + ni] += A[mi * n + ki] * B[ki * n + ni];
    }
  }

  double start, end;
  start = time_ms();
  dgemm(n, A, B, C);
  end = time_ms();

  double cpu_time_used = end - start;
  int count = 0;
  for (int i = 0; i < n * n; i++)
    if (C_golden[i] != C[i])
    {
      count++;
      // printf("c_golden is %lf,c is %lf \n",C_golden[i],C[i]);
    }
  if (count == 0)
    printf("GEMM 1 (row-col, B row-major) PASS!\n\n");
  else
    printf("GEMM 1 (row-col, B row-major)) NOT PASS!\n\n");
  printf("GEMM USE UNROLL: Time = %f ms\n", cpu_time_used);

  // for (int i = 0; i < n; i++) {
  //     for (int j = 0; j < n; j++) {
  //         printf("%6.2f ", C[i * n + j]);
  //     }
  //     printf("\n");
  // }

  free(A);
  free(B);
  free(C);
  return 0;
}
