#include "cudaImg.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <opencv2/opencv.hpp>


__global__ void getHist(unsigned char* dataIn, unsigned int* hist)
{

    int xdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int ydx = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    int tid = xdx + ydx*gridDim.x*blockDim.x; //全局ID

    atomicAdd(&hist[dataIn[tid]],1);

}

void getHist_gpu(unsigned char* dataIn,
                 unsigned int* hist,
                 dim3 tPerBlock,
                 dim3 bPerGrid)
{
    getHist<<<bPerGrid,tPerBlock>>>(dataIn,hist);
}