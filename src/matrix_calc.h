#ifndef MATRIX_CALC_H
#define MATRIX_CALC_H

// CUDA runtime
#include <cublas_v2.h>
#include "../public/matrix.h"
/*
* use cpp complier
#ifdef __cplusplus
extern "C" {
#endif
*/

/** kernel function:
 * Pay Attention:
 *   1 don't use reference variable, kernel can't access host memory directly
 *   2 matrix stored in column-major format
 */
void matrix_invert_host(Matrix host_input, Matrix host_output, int* pInfoArr);
void matrix_mul_host(Matrix host_A,Matrix host_B, Matrix host_output,cublasOperation_t transa=CUBLAS_OP_N,cublasOperation_t transb=CUBLAS_OP_N);


/*
 * Linear Regression:
 *    θ = Invert(X┬ X) X┬ Y
 * pInfoArr:
 *    a pointer to an integer array. array size == host_X.size
 *    Used to return the error when Inverse matrix!
 */
void LinearRegressionV2(Matrix &host_X,Matrix &host_Y,Matrix &host_ans, int* pInfoArr);



#endif
