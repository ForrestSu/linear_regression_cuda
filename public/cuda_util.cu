#include <stdio.h>
// CUDA runtime
#include <cublas_v2.h>
#include <cuda_runtime.h>
// helper functions
#include "helper_string.h"
#include "helper_cuda.h"
#include "device_functions.h"

#include "cuda_util.h"


/////////工具函数//////
void AllocDevice(Matrix& dev_a)
{
   dev_a.elements =NULL;
   int batchSize = dev_a.size;
   REAL** dev_ptr= (REAL **)malloc( batchSize * sizeof(REAL*));
   for(int i=0; i < batchSize;i++)
       checkCudaErrors(cudaMalloc((void**) &(dev_ptr[i]), dev_a.height * dev_a.width * sizeof(REAL*)));

   checkCudaErrors(cudaMalloc( (void**) &(dev_a.elements), batchSize * sizeof(REAL*)));
   checkCudaErrors(cudaMemcpy( dev_a.elements, dev_ptr , batchSize * sizeof(REAL*), cudaMemcpyHostToDevice));
   free(dev_ptr);
}
void FreeDevice(Matrix& dev_a)
{
   int batchSize = dev_a.size;
   REAL** host_ptr= (REAL **)malloc( batchSize * sizeof(REAL*));
   checkCudaErrors(cudaMemcpy(host_ptr, dev_a.elements , batchSize * sizeof(REAL*), cudaMemcpyDeviceToHost));

   for(int i=0; i < batchSize;i++)
        cudaFree( host_ptr[i]);

   checkCudaErrors(cudaFree(dev_a.elements));
   free(host_ptr);
}

void Copy2Device(Matrix& dev_a,Matrix& host_A)
{
    if(dev_a.size != host_A.size)
    {
        printf("error: can't copy!");
        return;
    }
    int batchSize = host_A.size;
    REAL** host_ptr= (REAL **)malloc( batchSize * sizeof(REAL*));
    checkCudaErrors(cudaMemcpy(host_ptr, dev_a.elements , batchSize * sizeof(REAL*), cudaMemcpyDeviceToHost));

    //ShowMatrixByRow2(host_A);
    for(int i=0; i < batchSize;i++)
    {
        // printf("host_ptr==%p\n ",host_ptr[i]);
        checkCudaErrors(cudaMemcpy(host_ptr[i], host_A.elements[i], dev_a.height * dev_a.width*sizeof(REAL), cudaMemcpyHostToDevice));
    }
    free(host_ptr);
}
void CopyBack2Host(Matrix& host_A,Matrix& dev_a)
{
    if(dev_a.size != host_A.size)
    {
        printf("error: can't copy to host!");
        return;
    }
    int batchSize = dev_a.size;
    REAL** dev_ptr= (REAL **)malloc( batchSize * sizeof(REAL*));
    checkCudaErrors(cudaMemcpy(dev_ptr, dev_a.elements , batchSize * sizeof(REAL*), cudaMemcpyDeviceToHost));
    for(int i=0; i < batchSize;i++)
    {
         checkCudaErrors(cudaMemcpy(host_A.elements[i], dev_ptr[i], dev_a.height * dev_a.width*sizeof(REAL), cudaMemcpyDeviceToHost));
    }
    free(dev_ptr);
}




