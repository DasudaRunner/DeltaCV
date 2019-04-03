#include "deltaCV/gpu/cudaImg.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <opencv2/opencv.hpp>

__global__ void getHist(unsigned char* dataIn, unsigned int* hist)
{

    int xdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int ydx = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    int tid = xdx + ydx*gridDim.x*blockDim.x;

    if(tid < 256)
    {
        hist[tid]=0;
    }
    __syncthreads();

    atomicAdd(&hist[dataIn[tid]],1);

}

void getHist_gpu(unsigned char* dataIn,
                 unsigned int* hist,
                 dim3 tPerBlock,
                 dim3 bPerGrid)
{
    getHist<<<bPerGrid,tPerBlock>>>(dataIn,hist);
}

//__global__ void getHist0(unsigned char* dataIn,
//                         unsigned int* hist,
//                         unsigned int* sum_ni,
//                         unsigned char* dataOut,
//                         short int imgRows,
//                         short int imgCols)
//{
//
//    int xdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
//    int ydx = threadIdx.y + __umul24(blockIdx.y, blockDim.y);
//
//    int tid = xdx + ydx*gridDim.x*blockDim.x;
//
//    int tid_block = threadIdx.x+blockDim.x*threadIdx.y;
//
//    unsigned char val = dataIn[tid];
//
//    if(blockIdx.x==0 && blockIdx.y==0 && tid_block<256)
//    {
//        hist[tid_block]=0;
//        sum_ni[tid_block]=0;
//    }
//    __syncthreads();
//
//    atomicAdd(&hist[val],1);
//
//    __syncthreads();
//
////    if((ydx<256) && (xdx<=ydx))
////    {
////        atomicAdd(&sum_ni[ydx],hist[xdx]);
////    }
////    atomicAdd(&sum_ni[1],1);
//
//    if(tid==0)
//    {
//        unsigned int tttt = hist[0];
//        sum_ni[1] = tttt;
//        atomicAdd(&sum_ni[0],tttt);
//    }
//
////    __syncthreads();
////
////    dataOut[tid] = (255*sum_ni[val])/(imgRows*imgCols);
//
//}
//
//void getHist0_gpu(unsigned char* dataIn,
//                  unsigned int* hist,
//                  unsigned int* sum_ni,
//                  unsigned char* dataOut,
//                  short int imgRows,
//                  short int imgCols,
//                 dim3 tPerBlock,
//                 dim3 bPerGrid)
//{
//    getHist0<<<bPerGrid,tPerBlock>>>(dataIn,hist,sum_ni,dataOut,imgRows,imgCols);
//}

__global__ void equalizeHistStep0(unsigned char* dataIn,
                                  unsigned int* hist,
                                  unsigned int* sum_ni)
{

    int xdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int ydx = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    int tid = xdx + ydx*gridDim.x*blockDim.x;

    if(tid < 256)
    {
        hist[tid]=0;
        sum_ni[tid]=0;
    }
    __syncthreads();

    atomicAdd(&hist[dataIn[tid]],1);
}

__global__ void equalizeHistStep1(unsigned int* hist,
                                  unsigned int* sum_ni)
{
    if(threadIdx.x<=blockIdx.x)
    {
        atomicAdd(&sum_ni[blockIdx.x],hist[threadIdx.x]);
    }
}

__global__ void equalizeHistStep2(unsigned char* dataIn,
                                  unsigned int* sum_ni,
                                  unsigned char* dataOut,
                                  short int imgRows,
                                  short int imgCols)
{
    int xdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int ydx = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    int tid = xdx + ydx*gridDim.x*blockDim.x;

    dataOut[tid] = (255*sum_ni[dataIn[tid]])/(imgRows*imgCols);
}
void equalizeHist_gpu(unsigned char* dataIn,
                  unsigned int* hist,
                  unsigned int* sum_ni,
                  unsigned char* dataOut,
                  short int imgRows,
                  short int imgCols,
                  dim3 tPerBlock,
                  dim3 bPerGrid)
{
    equalizeHistStep0<<<bPerGrid,tPerBlock>>>(dataIn,hist,sum_ni);

    dim3 tPerBlock_pi(256,1);
    dim3 bPerGrid_pi(256,1);

    equalizeHistStep1<<<bPerGrid_pi,tPerBlock_pi>>>(hist,sum_ni);

    equalizeHistStep2<<<bPerGrid,tPerBlock>>>(dataIn,sum_ni,dataOut,imgRows,imgCols);
}
