// includes, system
// helper functions
#include "matrix_calc.h"
#include "helper_cuda.h"
#include <cuda_runtime.h>
#include "device_functions.h"

#include "../public/cuda_util.h"

__device__ void CheckResultError(const int *arr, int size, const char* info)
{
    int iErrorCnt = 0;
    for (int i = 0; i < size; ++i)
    {
        if (arr[i] != 0)
        {
            ++iErrorCnt;
            printf("[Error] %s Fail, %d matrix's infoArr=%d\n", info, i, arr[i]);
        }
    }
    if(iErrorCnt>0) printf("BatchSize = %d, iErrorCnt=%d.\n", size, iErrorCnt);
}
//=========================================//
////////////  Matrix  Inverse  //////////////
//========================================//
__global__ void matrix_invert_dev(Matrix dev_in, Matrix dev_out, int* infoArr)
{
    cublasHandle_t hdl;
    cublasStatus_t status = cublasCreate(&hdl);
    int dimN = dev_in.height;
    //in fact lda(leading dimension) = dev_in.height, if stored in column-major format!
    int lda = dev_in.height;
    //in fact ldc(leading dimension) = dev_out.height
    int ldc = dev_out.height;
     //how many matrix need inverse
    int batchSize = dev_in.size;

    int *pivotArr;
    pivotArr = (int *)malloc(dimN * batchSize * sizeof(int));
    //infoArr = (int *) malloc(batchSize * sizeof(int));
    for (int i = 0; i < batchSize; i++)
        infoArr[i] = 0;
    // See
    // http://docs.nvidia.com/cuda/pdf/CUDA_Dynamic_Parallelism_Programming_Guide.pdf
    /*
     * hdl: cuBLAS library context
     * n : number of rows and columns of Aarray[i].
     * float *Aarray[]: array of pointer to Matrix
     * lda: leading dimension
     * pivotArr : array of size n x batchSize that contains the pivoting sequence of each factorization of Aarray[i] stored in a linear fashion.
     * infoArr: array of size batchSize that info(=infoArray[i]) contains the information of factorization of Aarray[i].
     * batchSize: number of pointers contained in A
     */
    status = emsCublasREALgetrfBatched(hdl, dimN, dev_in.elements, lda, pivotArr, infoArr, batchSize);
    __syncthreads();
    //check error
    if(status)printf("[warn] status = %d in rf!",status);
    {
        CheckResultError(infoArr,batchSize,"rf");
    }
    status = emsCublasREALgetriBatched(hdl, dimN, (const REAL**)dev_in.elements, lda, pivotArr, dev_out.elements, ldc, infoArr, batchSize);
    __syncthreads();
    //check error
    if(status)printf("[warn] status = %d in ri!",status);
    {
        CheckResultError(infoArr,batchSize,"ri");
    }
    //must call destroy for release resource
    free(pivotArr);
    //free(infoArr);
    cublasDestroy(hdl);
}


void matrix_invert_host(Matrix host_input, Matrix host_output, int* pInfoArr)
{
    int batchSize = host_input.size;
    int* d_infoArr = NULL;

    Matrix d_a = host_input;
    Matrix d_c = host_output;
    AllocDevice(d_a);
    AllocDevice(d_c);
    checkCudaErrors(cudaMalloc((void** ) &(d_infoArr), batchSize * sizeof(int)));

    Copy2Device(d_a, host_input);

    matrix_invert_dev<<<1,1>>>(d_a, d_c, d_infoArr);
    checkCudaErrors(cudaDeviceSynchronize());
    //copy back
    CopyBack2Host(host_output, d_c);
    if (pInfoArr != NULL)
        checkCudaErrors(cudaMemcpy(pInfoArr, d_infoArr, batchSize * sizeof(int), cudaMemcpyDeviceToHost));

    FreeDevice(d_a);
    FreeDevice(d_c);
    checkCudaErrors(cudaFree(d_infoArr));
}



//////////////////////////////////////////////////
////////////      matrix multiply     ////////////
//////////////////////////////////////////////////
__global__ void matrix_mul_dev(Matrix dev_a,Matrix dev_b, Matrix dev_c,cublasOperation_t transa,cublasOperation_t transb)
{
    cublasHandle_t hdl;
    cublasStatus_t status= cublasCreate(&hdl);
    int batchSize = (dev_a.size <= dev_b.size) ? dev_a.size:dev_b.size ;
    /**
     * C = α op ( A ) op ( B ) + β C
     */
    REAL alpha = 1.0;
    REAL beta = 0.0;
    /**
     * leading dimension of two-dimensional array used to store matrix A.
     */
    int lda = dev_a.height;
    int ldb = dev_b.height;
    int ldc = dev_c.height;
    /**
     * m: number of rows of matrix op(A) and C.
     * n: number of columns of matrix op(B) and C.
     * k: number of columns of op(A) and rows of op(B).
     */
    int m = dev_c.height;
    int n = dev_c.width;
    int k = (transb == CUBLAS_OP_T) ? dev_b.width : dev_b.height;
    status= emscublasREALgemmBatched(hdl,transa,transb, m , n, k, &alpha, (const REAL **) dev_a.elements ,lda, (const REAL **)dev_b.elements , ldb , &beta, dev_c.elements, ldc, batchSize);
    __syncthreads();

    if(status)printf("status = %d in sge!",status);
    cublasDestroy(hdl);
}


void matrix_mul_host(Matrix host_A,Matrix host_B, Matrix host_output,cublasOperation_t transa,cublasOperation_t transb)
{
    Matrix dev_a, dev_b,dev_c;
    dev_a = host_A;
    dev_b = host_B;
    dev_c = host_output;

    AllocDevice(dev_a);
    AllocDevice(dev_b);
    AllocDevice(dev_c);

    Copy2Device(dev_a,host_A);
    Copy2Device(dev_b,host_B);

    matrix_mul_dev<<<1,1>>> (dev_a,dev_b,dev_c,transa,transb);

    checkCudaErrors(cudaDeviceSynchronize());
    //copy back
    CopyBack2Host(host_output,dev_c);

    FreeDevice(dev_a);
    FreeDevice(dev_b);
    FreeDevice(dev_c);

}

/*
 * Linear Regression:
 *    θ = Invert(X┬ X) X┬ Y
 * pInfoArr:
 *    a pointer to an integer array. array size == host_X.size
 *    Used to return the error when Inverse matrix!
 */
void LinearRegressionV2(Matrix &host_X, Matrix &host_Y, Matrix &host_ans, int* pInfoArr)
{
    int batchSize = host_X.size;
    int* dev_infoArr = NULL;

    Matrix dev_X = host_X;
    Matrix dev_XT_X = host_X;
    dev_XT_X.height = host_X.width;
    Matrix dev_invX = dev_XT_X;

    AllocDevice(dev_X);
    AllocDevice(dev_XT_X);
    AllocDevice(dev_invX);
    checkCudaErrors(cudaMalloc((void** ) &(dev_infoArr), batchSize * sizeof(int)));
    Copy2Device(dev_X, host_X);

    matrix_mul_dev<<<1,1>>> (dev_X, dev_X, dev_XT_X, CUBLAS_OP_T, CUBLAS_OP_N);
    checkCudaErrors(cudaDeviceSynchronize());

    matrix_invert_dev<<<1,1>>>(dev_XT_X, dev_invX, dev_infoArr);
    checkCudaErrors(cudaDeviceSynchronize());

    if (pInfoArr != NULL)
        checkCudaErrors(cudaMemcpy(pInfoArr, dev_infoArr, batchSize * sizeof(int), cudaMemcpyDeviceToHost));

    //release 1 temporary matrix,After (dev_X dev_invX) in Device memory
    FreeDevice(dev_XT_X);
	checkCudaErrors(cudaFree(dev_infoArr));

    Matrix dev_3X;
    dev_3X.width = dev_X.height;
    dev_3X.height = dev_X.width;
    dev_3X.size = dev_X.size;
    AllocDevice(dev_3X);

    matrix_mul_dev<<<1,1>>> (dev_invX, dev_X, dev_3X, CUBLAS_OP_N, CUBLAS_OP_T);
    checkCudaErrors(cudaDeviceSynchronize());
    //release 2 temp matrix, After only dev_3X in Device memory
    FreeDevice(dev_invX);
    FreeDevice(dev_X);

    Matrix dev_Y = host_Y;
    Matrix dev_ans = host_ans;
    AllocDevice(dev_Y);
    AllocDevice(dev_ans);
    Copy2Device(dev_Y, host_Y);

    matrix_mul_dev<<<1,1>>> (dev_3X, dev_Y, dev_ans , CUBLAS_OP_N, CUBLAS_OP_N);
    checkCudaErrors(cudaDeviceSynchronize());

    CopyBack2Host(host_ans, dev_ans);

    FreeDevice(dev_3X);
    FreeDevice(dev_Y);
    FreeDevice(dev_ans);
}

