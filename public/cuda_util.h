#ifndef _CUDA_UTIL_H
#define _CUDA_UTIL_H

#include "matrix.h"
/*
#ifdef __cplusplus
extern "C" {
#endif
*/
void AllocDevice(Matrix& dev_a);
void FreeDevice(Matrix& dev_a);
void Copy2Device(Matrix& dev_a,Matrix& host_A);
void CopyBack2Host(Matrix& host_A,Matrix& dev_a);

/*
#ifdef __cplusplus
}
#endif
*/

#endif
