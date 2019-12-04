#include "matrix_util.h"
#include "util.h"
#include <iostream>
#include <fstream>
#include <stdio.h>

void InitialMatrix(Matrix &mat, int row, int col, int ibatchSize)
{
    mat.width = col;
    mat.height = row;
    mat.size = ibatchSize;
    mat.elements = (REAL**) malloc(mat.size * sizeof(REAL*));
    for (int batchSize = 0; batchSize < mat.size; batchSize++)
    {
        mat.elements[batchSize] = (REAL*) malloc(mat.height * mat.width * sizeof(REAL));
        for (int i = 0; i < mat.width * mat.height; ++i)
            mat.elements[batchSize][i] = 0;
    }
}

void FreeMatrix(Matrix &mat)
{
    for (int batchSize = 0; batchSize < mat.size; batchSize++)
    {
        free(mat.elements[batchSize]);
    }
    free(mat.elements);
}


////////////////////////////////////
/////   Test Code : for debug  /////
///////////////////////////////////

/**
 * print ibatchSize matrices at most! (human read format)
 */
void PrintMatrixGroup(Matrix &mat,int ibatchSize)
{
    int showCnt= std::min(ibatchSize,mat.size);
    printf("Will Show [%d] Matrix!\n",mat.size);
    for(int batchSize=0;batchSize<showCnt;batchSize++)
    {
        printf("\n[%d]Matrix(%d,%d)-->\n",batchSize, mat.height,mat.width);
        for(int i=0;i<mat.height;i++ )
        {
            for(int j=0;j<mat.width;j++ )
            {
                printf("%f\t", mat.elements[batchSize][ j * mat.height + i ]);
            }
            printf("\n");
        }
        printf("<--\n");
    }
}

/**
 * output Linear Regression to file!
 */
void WriteRegressionResult(std::map<std::string,std::string>& result, int type)
{
    //code,lag,time,threshold,voi_0,voi_1,voi_2,voi_3...,oir_0,oir_1,oir_2,oir_3...,mpb,price_change
    std::string curDate = StrUtil::itos(TimeUtil::CurDate());
    std::string file = "/home/toptrade/emslogs/cudacsv/";
    FileUtil::create_dir(file.c_str());
    file = file + ((type == 1) ? "stock" : "future");
    file = file + "_coefficient" + curDate + ".csv";
    std::ofstream fout(file,std::ios::app);
    fout.setf(std::ios::fixed, std::ios::floatfield);// set fixed mode, 以小数点表示浮点数
    fout.precision(8);
    for(auto it = result.begin(); it!=result.end(); ++it)
    {
        fout<<it->first <<"," << it->second <<"\n";
    }
    fout << "\n";
    fout.close();
}

void WriteOneFeature2File(Matrix &X,Matrix &Y, const int batchIndex, const std::string &filtCode, int idalay)
{
    std::string curDateTime = StrUtil::itos(TimeUtil::CurDate()) + "-"+ StrUtil::itos(TimeUtil::CurTime());
    std::string file = "/home/toptrade/emslogs/cudacsv/";
    FileUtil::create_dir(file.c_str());
    file = file+"cuda_feature_"+curDateTime +"(" + StrUtil::itos(idalay) + ").csv";
    std::ofstream fout(file, std::ios::app);
    fout.setf(std::ios::fixed, std::ios::floatfield);
    fout.precision(8);
    fout<<"Matrix(row="<<X.height<<",col=" << X.width <<"),total:"<< X.size<<", Code:" <<filtCode<<"\n";
    fout<<"voi_0,voi_1,voi_2,voi_3...,oir_0,oir_1,oir_2,oir_3...,mpb,price_change,Y\n";
    if (batchIndex < X.size)
    {
        for (int i = 0; i < X.height; i++)
        {
            for (int j = 0; j < X.width; j++)
            {
                fout << X.elements[batchIndex][j * X.height + i] << ",";
            }
            //price change
            fout << Y.elements[batchIndex][i] << "\n";
        }
        fout << "\n";
    }
    fout.close();
}

