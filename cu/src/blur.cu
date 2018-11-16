#include "cudaImg.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <opencv2/opencv.hpp>

__global__ void guassianBlur3(unsigned char* dataIn,
                     unsigned char* dataOut,
                      short int imgRows,
                      short int imgCols)
{
    int xdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int ydx = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    if(xdx>0 && xdx<imgCols-1 && ydx>0 && ydx<imgRows-1)
    {
        dataOut[xdx + ydx * imgCols] =
                (dataIn[(xdx-1)+(ydx-1)*imgCols] + dataIn[(xdx)+(ydx-1)*imgCols]*2 + dataIn[(xdx+1)+(ydx-1)*imgCols]+
                        dataIn[(xdx-1)+(ydx)*imgCols]*2 + dataIn[(xdx)+(ydx)*imgCols]*4 + dataIn[(xdx+1)+(ydx)*imgCols]*2 +
                        dataIn[(xdx-1)+(ydx+1)*imgCols] + dataIn[(xdx)+(ydx+1)*imgCols]*2 + dataIn[(xdx+1)+(ydx+1)*imgCols])/16;
    }

}

//__global__ void guassianBlur5(cudaTextureObject_t dataIn,
//                      unsigned char* dataOut,
//                      short int imgRows,
//                      short int imgCols)
//{
//    int xdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
//    int ydx = threadIdx.y + __umul24(blockIdx.y, blockDim.y);
//
//    dataOut[xdx+imgCols*ydx] =
//            (tex2D(dataIn,xdx-2,ydx-2) + tex2D(dataIn,xdx-1,ydx-2)*4 + tex2D(dataIn,xdx,ydx-2)*7 + tex2D(dataIn,xdx+1,ydx-2)*4 + tex2D(dataIn,xdx+2,ydx-2) +
//            tex2D(dataIn,xdx-2,ydx-1)*4 + tex2D(dataIn,xdx-1,ydx-1)*16 + tex2D(dataIn,xdx,ydx-1)*26 + tex2D(dataIn,xdx+1,ydx-1)*16 + tex2D(dataIn,xdx+2,ydx-1)*4 +
//            tex2D(dataIn,xdx-2,ydx)*7 + tex2D(dataIn,xdx-1,ydx)*26 + tex2D(dataIn,xdx,ydx)*41 + tex2D(dataIn,xdx+1,ydx)*26 + tex2D(dataIn,xdx+2,ydx)*7 +
//            tex2D(dataIn,xdx-2,ydx+1)*4 + tex2D(dataIn,xdx-1,ydx+1)*16 + tex2D(dataIn,xdx,ydx+1)*26 + tex2D(dataIn,xdx+1,ydx+1)*16 + tex2D(dataIn,xdx+2,ydx+1)*4 +
//            tex2D(dataIn,xdx-2,ydx+2) + tex2D(dataIn,xdx-1,ydx+2)*4 + tex2D(dataIn,xdx,ydx+2)*7 + tex2D(dataIn,xdx+1,ydx+2)*4 + tex2D(dataIn,xdx+2,ydx+2))/273;
//
//}

void guassianBlur3_gpu(unsigned char* dataIn,
                      unsigned char* dataOut,
                      short int imgRows,
                      short int imgCols,
                      dim3 tPerBlock,
                      dim3 bPerGrid)
{
        guassianBlur3<<<bPerGrid,tPerBlock>>>(dataIn,dataOut,imgRows,imgCols);

}