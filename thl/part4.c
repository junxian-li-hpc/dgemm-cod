#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>

#define UNROLL 4

void dgemm(int n, double *A, double *B, double *C)
{
    memset(C, 0, sizeof(double) * n * n); // 初始化C矩阵为0

    // 遍历矩阵的行（mi）和列（ki）
    for (int mi = 0; mi < n; mi++)
    {
        for (int ki = 0; ki < n; ki++)
        {
            double aval = A[mi * n + ki];
            __m256d aavx2 = _mm256_set1_pd(aval); // 将 aval 广播到所有向量元素中

            int j = 0;
            // 使用 UNROLL 展开处理C和B矩阵，处理4个元素的倍数部分
            for (; j <= n - UNROLL * 4; j += UNROLL * 4)
            {
                // 每次处理 4 个连续的元素（共16个double值）
                __m256d bavx0 = _mm256_loadu_pd(&B[ki * n + j]);
                __m256d bavx1 = _mm256_loadu_pd(&B[ki * n + j + 4]);
                __m256d bavx2 = _mm256_loadu_pd(&B[ki * n + j + 8]);
                __m256d bavx3 = _mm256_loadu_pd(&B[ki * n + j + 12]);

                __m256d cavx0 = _mm256_loadu_pd(&C[mi * n + j]);
                __m256d cavx1 = _mm256_loadu_pd(&C[mi * n + j + 4]);
                __m256d cavx2 = _mm256_loadu_pd(&C[mi * n + j + 8]);
                __m256d cavx3 = _mm256_loadu_pd(&C[mi * n + j + 12]);

                cavx0 = _mm256_fmadd_pd(aavx2, bavx0, cavx0); // C += A * B
                cavx1 = _mm256_fmadd_pd(aavx2, bavx1, cavx1);
                cavx2 = _mm256_fmadd_pd(aavx2, bavx2, cavx2);
                cavx3 = _mm256_fmadd_pd(aavx2, bavx3, cavx3);

                _mm256_storeu_pd(&C[mi * n + j], cavx0); // 存储回 C
                _mm256_storeu_pd(&C[mi * n + j + 4], cavx1);
                _mm256_storeu_pd(&C[mi * n + j + 8], cavx2);
                _mm256_storeu_pd(&C[mi * n + j + 12], cavx3);
            }

            // 处理剩余的部分（不足16个double的部分）
            for (; j < n; j++)
            {
                C[mi * n + j] += aval * B[ki * n + j];
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

    clock_t start = clock();
    dgemm(n, A, B, C);
    clock_t end = clock();

    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
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
    printf("GEMM USE UNROLL: Time = %f seconds\n", cpu_time_used);

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
