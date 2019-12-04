/*
 * matrix_util.h
 *
 *  Created on: 2017年7月10日
 *      Author: sunquan
 *       Email: sunquana@gmail.com
 */
#ifndef MATRIX_UTIL_H_
#define MATRIX_UTIL_H_

#include <vector>
#include <string>
#include <map>
#include "matrix.h"


void InitialMatrix(Matrix &mat,int row,int col,int ibatchSize);
void FreeMatrix(Matrix &mat);


//test code
void PrintMatrixGroup(Matrix &mat,int ibatchSize);
void WriteRegressionResult(std::map<std::string,std::string>& result, int type);
// print one stock-feature
void WriteOneFeature2File(Matrix &X, Matrix &Y, const int batchIndex,  const std::string &filtCode, int idalay);


#endif /* MATRIX_UTIL_H_ */
