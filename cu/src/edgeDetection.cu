#include "cudaImg.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

__global__ void sobel(unsigned char* dataIn,
                      unsigned char* dataOut,
                      short int imgRows,
                      short int imgCols)
{
    int xdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int ydx = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    short int Gx=0;
    short int Gy=0;

    if(xdx>0 && xdx<imgCols-1 && ydx>0 && ydx<imgRows-1)
    {
        Gx = dataIn[(xdx-1)+(ydx-1)*imgCols] - dataIn[(xdx+1)+(ydx-1)*imgCols] +
                dataIn[(xdx-1)+ydx*imgCols]*2 - dataIn[(xdx+1)+ydx*imgCols]*2 +
                dataIn[(xdx-1)+(ydx+1)*imgCols] - dataIn[(xdx+1)+(ydx+1)*imgCols];

        Gy = dataIn[(xdx-1)+(ydx-1)*imgCols] + dataIn[xdx+(ydx-1)*imgCols]*2 + dataIn[(xdx+1)+(ydx-1)*imgCols] -
                dataIn[(xdx-1)+(ydx+1)*imgCols] - dataIn[xdx+(ydx+1)*imgCols]*2 - dataIn[(xdx+1)+(ydx+1)*imgCols];

        if(Gx<0)
            Gx = -Gx;
        if(Gy<0)
            Gy = -Gy;

        dataOut[xdx + ydx*imgCols] = (Gx+Gx)/2;

//        if ((Gx+Gx)/2 > 100)
//        {
//            dataOut[tid]=255;
//        } else{
//            dataOut[tid]=0;
//        }
    }

}

__global__ void scharr(unsigned char* dataIn,
                       unsigned char* dataOut,
                       short int imgRows,
                       short int imgCols)
{
    int xdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int ydx = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    short int Gx=0;
    short int Gy=0;

    if(xdx>0 && xdx<imgCols-1 && ydx>0 && ydx<imgRows-1)
    {
        Gx = dataIn[(xdx-1)+(ydx-1)*imgCols]*3 - dataIn[(xdx+1)+(ydx-1)*imgCols]*3 +
             dataIn[(xdx-1)+ydx*imgCols]*10 - dataIn[(xdx+1)+ydx*imgCols]*10 +
             dataIn[(xdx-1)+(ydx+1)*imgCols]*3 - dataIn[(xdx+1)+(ydx+1)*imgCols]*3;

        Gy = dataIn[(xdx-1)+(ydx-1)*imgCols]*3 + dataIn[xdx+(ydx-1)*imgCols]*10 + dataIn[(xdx+1)+(ydx-1)*imgCols]*3 -
             dataIn[(xdx-1)+(ydx+1)*imgCols]*3 - dataIn[xdx+(ydx+1)*imgCols]*10 - dataIn[(xdx+1)+(ydx+1)*imgCols]*3;

        if(Gx<0)
            Gx = -Gx;
        if(Gy<0)
            Gy = -Gy;

        dataOut[xdx + ydx*imgCols] = (Gx+Gx)/2;

//        if ((Gx+Gx)/2 > 100)
//        {
//            dataOut[tid]=255;
//        } else{
//            dataOut[tid]=0;
//        }
    }

}

void sobel_gpu(unsigned char* dataIn,
               unsigned char* dataOut,
               short int imgRows,
               short int imgCols,
               dim3 tPerBlock,
               dim3 bPerGrid)
{
    sobel<<<bPerGrid,tPerBlock>>>(dataIn,dataOut,imgRows,imgCols);
}

void scharr_gpu(unsigned char* dataIn,
                unsigned char* dataOut,
                short int imgRows,
                short int imgCols,
               dim3 tPerBlock,
               dim3 bPerGrid)
{
    scharr<<<bPerGrid,tPerBlock>>>(dataIn,dataOut,imgRows,imgCols);
}