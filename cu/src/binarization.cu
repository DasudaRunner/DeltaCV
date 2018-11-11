#include "cudaImg.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h> //包含了预定义的变量

/*
 * Compare 'threshold()' funciton in OpenCV
 * When:
 *      thresholdMin = thresholdMax+1 and valMin = 0  ==> THRESH_BINARY
 *      thresholdMin = thresholdMax+1 and valMax = 0  ==> THRESH_BINARY_INV
 *      thresholdMax = valMax and thresholdMin = 0  ==> THRESH_TRUNC
 *      thresholdMax = 255 and valMin = 0  ==> THRESH_TOZERO
 *      thresholdMin = 0 and valMax = 0  ==> THRESH_TOZERO_INV
 */

__global__ void thresholdBinarization(unsigned char* dataIn,
                                      unsigned char* dataOut,
                                      short int imgRows,
                                      short int imgCols,
                                      unsigned char thresholdMin,
                                      unsigned char thresholdMax,
                                      unsigned char valMin,
                                      unsigned char valMax)
{
    int xIndex = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int yIndex = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    int tid = __umul24(yIndex,imgCols)+xIndex;

    unsigned char val=dataIn[tid];
    unsigned char res = val;

    if(xIndex < imgCols && yIndex < imgRows)
    {

        if(val>thresholdMax)
        {
            res = valMax;
        }

        if(val<thresholdMin)
        {
            res = valMin;
        }

        dataOut[tid] = res;
    }
}

void thresholdBinarization_gpu(unsigned char* dataIn,
                               unsigned char* dataOut,
                               short int imgRows,
                               short int imgCols,
                               unsigned char thresholdMin,
                               unsigned char thresholdMax,
                               unsigned char valMin,
                               unsigned char valMax,
                               dim3 tPerBlock,
                               dim3 bPerGrid)
{
    thresholdBinarization<<<bPerGrid,tPerBlock>>>(dataIn,dataOut,imgRows,imgCols,thresholdMin,thresholdMax,valMin,valMax);
}