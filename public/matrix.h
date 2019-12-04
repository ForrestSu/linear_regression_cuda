#ifndef _MATRIX_H_
#define _MATRIX_H_

#define DOUBLE_PRECISION

#ifdef DOUBLE_PRECISION
typedef double REAL;
#define emsCublasREALgetrfBatched  cublasDgetrfBatched
#define emsCublasREALgetriBatched  cublasDgetriBatched
#define emscublasREALgemmBatched   cublasDgemmBatched
#else
typedef float REAL;
#define emsCublasREALgetrfBatched  cublasSgetrfBatched
#define emsCublasREALgetriBatched  cublasSgetriBatched
#define emscublasREALgemmBatched   cublasSgemmBatched
#endif

typedef struct {
    int width;  //col
    int height; //rows
    REAL** elements; /*array of pointer to Matrix,Each matrix stored in column-major format! **/
    int size;   //the size of a batch of matrices
} Matrix;


#endif
