//
// Created by dasuda on 18-10-27.
//
#include "cudaImg.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h> //包含了预定义的变量

__global__ void RGB2Gray(uchar3* dataIn,
                         unsigned char* dataOut,
                         int imgRows,
                         int imgCols)
{
    int xIndex = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int yIndex = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    int tid = __umul24(yIndex,imgCols)+xIndex;

    if(xIndex < imgCols && yIndex < imgRows)
    {
        uchar3 rgb = dataIn[tid];
//        dataOut[tid] = 0.299f * rgb.z + 0.587f * rgb.y + 0.114f * rgb.x;
        dataOut[tid] = (76 * rgb.z + 150 * rgb.y + 30 * rgb.x)>>8; //0.12ms 低精度
    }

}

void RGB2GRAY_gpu(uchar3* dataIn,
                  unsigned char* dataOut,
                  int imgRows,
                  int imgCols,
                  dim3 tPerBlock,
                  dim3 bPerGrid)
{
    RGB2Gray<<<bPerGrid,tPerBlock>>>(dataIn,dataOut,imgRows,imgCols);
}

/*
 * 在原图上（RGB空间）直接利用HSV的V通道进行约束
 */

__global__ void RGB2HSV_V(uchar3* dataIn,
                        uchar3* dataOut,
                         int imgRows,
                         int imgCols,
                          short int minVal,
                          short int maxVal)
{
    int xIndex = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int yIndex = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    int tid = yIndex*imgCols+xIndex;

    char maxPixel;
    uchar3 zeroPixel;
    zeroPixel.x=0;
    zeroPixel.y=0;
    zeroPixel.z=0;

    if(xIndex < imgCols && yIndex < imgRows)
    {
        uchar3 rgb = dataIn[tid];

        maxPixel = rgb.y;

        if(rgb.z > rgb.y)
        {
            maxPixel = rgb.z;
        }

        if(rgb.x > maxPixel)
        {
            maxPixel = rgb.x;
        }
        dataOut[tid] = (maxPixel<minVal || maxPixel>maxVal) ? zeroPixel:rgb;
    }
}

void RGB2HSV_V_gpu(uchar3* dataIn,
                 uchar3* dataOut,
                  int imgRows,
                  int imgCols,
                  short int minVal,
                  short int maxVal,
                  dim3 tPerBlock,
                  dim3 bPerGrid)
{
    RGB2HSV_V<<<bPerGrid,tPerBlock>>>(dataIn,dataOut,imgRows,imgCols,minVal,maxVal);
}