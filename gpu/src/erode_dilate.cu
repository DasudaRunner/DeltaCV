#include "deltaCV/gpu/cudaImg.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <opencv2/opencv.hpp>

//腐蚀
__global__ void erode(unsigned char* dataIn,
                      unsigned char* dataOut,
                       short int imgRows,
                       short int imgCols,
                       short int erodeElementRows,
                       short int erodeElementCols)
{
    int xdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int ydx = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    int tid = xdx + ydx * imgCols;

    char val = dataIn[tid];

    dataOut[tid] = dataIn[tid];

    if(xdx > erodeElementCols-1 && xdx < imgCols-erodeElementCols
            && ydx>erodeElementRows && ydx < imgRows-erodeElementRows)
    {
        for (int i = -erodeElementRows; i < erodeElementRows+1; ++i) { //行
            for (int j = -erodeElementCols; j < erodeElementCols+1; ++j) { //列
                char temp_val = dataIn[(ydx+i)*imgCols+(xdx+j)];
                if(temp_val < val)
                {
                    dataOut[tid] = temp_val;
                }
            }
        }
    }

}

//膨胀
__global__ void dilate(unsigned char* dataIn,
                       unsigned char* dataOut,
                       short int imgRows,
                       short int imgCols,
                       short int dilateElementRows,
                       short int dilateElementCols)
{
    int xdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int ydx = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    int tid = xdx + ydx * imgCols;

    char val = dataIn[tid];

    dataOut[tid] = dataIn[tid];

    if(xdx > dilateElementCols-1 && xdx < imgCols-dilateElementCols
       && ydx>dilateElementRows && ydx < imgRows-dilateElementRows)
    {
        for (int i = -dilateElementRows; i < dilateElementRows+1; ++i) { //行
            for (int j = -dilateElementCols; j < dilateElementCols+1; ++j) { //列
                char temp_val = dataIn[(ydx+i)*imgCols+(xdx+j)];
                if(temp_val > val)
                {
                    dataOut[tid] = temp_val;
                }
            }
        }
    }

}

void erode_gpu( unsigned char* dataIn,
                unsigned char* dataOut,
                short int imgRows,
                short int imgCols,
               cv::Size erodeSize,
               dim3 tPerBlock,
               dim3 bPerGrid)
{
    erode<<<bPerGrid,tPerBlock>>>(dataIn,dataOut,imgRows,imgCols,(erodeSize.height-1)/2,(erodeSize.width-1)/2);
}

void dilate_gpu(unsigned char* dataIn,
                unsigned char* dataOut,
                short int imgRows,
                short int imgCols,
               cv::Size dilateSize,
               dim3 tPerBlock,
               dim3 bPerGrid)
{
    dilate<<<bPerGrid,tPerBlock>>>(dataIn,dataOut,imgRows,imgCols,(dilateSize.height-1)/2,(dilateSize.width-1)/2);
}
