#include <stdio.h>
#include <sys/time.h>
#include "dgemm.h"

static double time_ms(void)
{
	struct timeval tm;
	double ms = 0.0;

	(void)gettimeofday(&tm, NULL);
	ms = tm.tv_sec * 1000.0;
	ms += tm.tv_usec / 1000.0;

	return ms;
}

//#define N 1024
#define N 1024
static double A[N*N], B[N*N], C[N*N], D[N*N], E[N*N], F[N*N];

static void data_init(int NN, double *AA, double *BB, double *CC)
{
	for (size_t i = 0; i < NN; ++i) {
		for (size_t j = 0; j < NN; ++j) {
			AA[i*NN + j] = i + NN * j;
			BB[i*NN + j] = i + NN * j;
			CC[i*NN + j] = i + NN * j;
		}
	}
}

static int data_check(int NN, double *A, double *B, int line)
{
	for (int y = 0; y < NN; y++) {
		for (int x = 0; x < NN; x++) {
			int idx = x + y * NN;
			if (A[idx] != B[idx]) {
				fprintf(stderr, "Line %d, A[%d][%d] = %f, B[%d][%d] = %f\n",
						line, x, y, A[idx], x, y, B[idx]);
				return -1;
			}
		}
	}

	return 0;
}

#define CHECK(NN, A, B) do {	\
	int ret = data_check(NN, A, B, __LINE__);	\
	if (ret < 0)	\
		return -1;	\
} while (0)

int main()
{
	double start, end;

	data_init(N, A, B, C);
	start = time_ms();
	dgemm_c(N, A, B, C);
	end = time_ms();
	printf("dgemm_c spends %f ms\n", end - start);

	data_init(N, D, E, F);
	start = time_ms();
	dgemm_avx(N, D, E, F);
	end = time_ms();
	printf("dgemm_avx spends %f ms\n", end - start);

	CHECK(N, C, F);

	data_init(N, D, E, F);
	start = time_ms();
	dgemm_avx_unroll(N, D, E, F);
	end = time_ms();
	printf("dgemm_avx_unroll spends %f ms\n", end - start);

	CHECK(N, C, F);

	data_init(N, D, E, F);
	start = time_ms();
	dgemm_avx_unroll_blk(N, D, E, F);
	end = time_ms();
	printf("dgemm_avx_unroll_blk spends %f ms\n", end - start);

	CHECK(N, C, F);

	data_init(N, D, E, F);
	start = time_ms();
	dgemm_avx_unroll_blk_omp(N, D, E, F);
	end = time_ms();
	printf("dgemm_avx_unroll_blk_omp spends %f ms\n", end - start);


	return 0;
}


