#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <mkl_cblas.h>

#define N 1009
#define REPEATS 25

// define a test element to be output
#define NOUT 1
#define outputElement NOUT*N+NOUT

void mult(float*, float*, float*);

int main(void) {
  float *m1, *m2, *res;
  int   matrixByte = N*N*sizeof(float);
  int i,j,k;
  double timer[8];

  printf("\n\n R U N N I N G ... ... \n\n");
  printf("N=%d\n", N);

  // malloc everybody
  m1 = malloc(matrixByte);
  m2 = malloc(matrixByte);
  res = malloc(matrixByte);

  // initialise
  for (i=0; i<N*N; i++) {
    m1[i] = (float) rand()/(float) RAND_MAX;
    m2[i] = (float) rand()/(float) RAND_MAX;
    res[i] = 0.0;
  }
  /* DEBUG
  printf("m1\n");
  for (i=0; i<N*N; i++) {
    printf("%f ",m1[i]);
  }
  printf("\n");
  printf("m2\n");
  for (i=0; i<N*N; i++) {
    printf("%f ",m2[i]);
  }
  printf("\n");
  */

  // mat mat mult
  // repeat a number of times
  int repeat;
  for (repeat=0; repeat<REPEATS; repeat++) {
    printf("Run %d\n", repeat);
    // reset res
    for (i=0; i<N*N; i++) {
      res[i] = 0.0;
    }
    timer[4] = omp_get_wtime();
    //    mult(res, m1, m2);

    // mat mat mult LIBRARY
    // float => single precision -> sgemm
    // sgemm is very generalised call sgemm. from `man sgemm`
    // sgemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    // to form C = alpha*A*B + beta*C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N,N,N, 1.0, m1, N, m2, N, 0.0, res,N);
    timer[5] = omp_get_wtime();
    // output a val to avoid compiler removing
    printf("res[outputElement] = %f\n",res[outputElement]); 
    printf("MAT MAT MULT: %f milliseconds\n\n", 1000.0*(timer[5]-timer[4]));
  }
}

void mult(float *res, float *m1, float *m2) {
  int i,j,k;
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) { 
	for (k=0; k<N; k++) {
	  int ij = i*N+j;
	  int kj = k*N+j;
	  int ik = i*N+k;
	  //	  res[i][j] += m1[i][k] * m2[k][j];
	  //DBG printf("res (%d) will be increased from %f by %f * %f to...", ij, res[ij], m1[ik], m2[kj]);
	  res[ij] += m1[ik] * m2[kj];
	  //DBG printf("  %f\n", res[ij]);
	}
      }
    }
}
