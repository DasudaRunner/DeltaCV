#include "deltaCV/gpu/cudaImg.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h> //包含了预定义的变量

/*
 * Compare 'threshold()' funciton in OpenCV
 * When:
 *      thresholdMin = thresholdMax and valMin = 0  ==> THRESH_BINARY
 *      thresholdMin = thresholdMax and valMax = 0  ==> THRESH_BINARY_INV
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

        if(val<=thresholdMin)
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

__global__ void thresholdBinarization_inner(unsigned char* dataIn,
                                            unsigned char* dataOut,
                                            short int imgRows,
                                            short int imgCols,
                                            int* thres)
{
    int xIndex = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int yIndex = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    int tid = __umul24(yIndex,imgCols)+xIndex;

    unsigned char val = dataIn[tid];

    if(xIndex < imgCols && yIndex < imgRows)
    {
        dataOut[tid] = 255-(val<=thres[0])*255;
    }
}

__global__ void tree_max(float* varance,int* thres)
{
        __shared__ float var[256];
        __shared__ int varId[256]; // 512*4/1024 = 2KB
        varId[threadIdx.x] = threadIdx.x;
        var[threadIdx.x] = varance[threadIdx.x];
        for (int i = 1; i < 256; i*=2)
        {
            if(threadIdx.x%(2*i)==0)
            {
                if(var[threadIdx.x]<var[threadIdx.x+i])
                {
                    var[threadIdx.x] = var[threadIdx.x+i];
                    varId[threadIdx.x] = varId[threadIdx.x+i];
                }
            }
            __syncthreads();
        }
        if(threadIdx.x == 0)
        {
            thres[0] = varId[0];
        }

}
/*
 * w_A:前景像素级占的比例，Pi的和
 * u_A:前景像素级的均值，i×Pi的和除以w_A
 */
__global__ void ostu(unsigned int* hist,
                     float* sum_Pi,
                     float* sum_i_Pi,
                     float* u_0,
                     float* varance,
                     short int imgRows,
                     short int imgCols)
{

    //清空相关数组，清空上一次的计算结果
    if(blockIdx.x == 0)
    {
        sum_Pi[threadIdx.x] = 0.0f;
        sum_i_Pi[threadIdx.x] = 0.0f;
        if(threadIdx.x==0)
        {
            u_0[0] = 0.0f;
        }
    }
    __syncthreads();

    //计算整幅图的平均灰度、前景的概率、前景的平均灰度值
    unsigned int current_val = hist[threadIdx.x];
    if(blockIdx.x==0)
    {
        atomicAdd(&u_0[0],current_val*threadIdx.x);//sum(i*Pi)+sum(j*Pj)
    }
    else
    {
        if(threadIdx.x < blockIdx.x)
        {
            atomicAdd(&sum_Pi[blockIdx.x-1],current_val);//sum()
            atomicAdd(&sum_i_Pi[blockIdx.x-1],current_val*threadIdx.x);//sum(i*Pi)
        }
    }
    __syncthreads();
    //now we get sum_Pi[256] and sum_i_Pi[256] and w_0
    //下面开始计算类间方差
    int imgSize = imgRows*imgCols;
    if(blockIdx.x>0)
    {
        float f_sum_pi = sum_Pi[blockIdx.x-1]/imgSize;
        float f_sum_pj = 1-f_sum_pi;
        if(f_sum_pj==0)
        {
            varance[blockIdx.x-1]=0;
        }else
        {
            float temp = (u_0[0]/imgSize-sum_i_Pi[blockIdx.x-1]/(f_sum_pi*imgSize));
            varance[blockIdx.x-1] = temp*temp*f_sum_pi/f_sum_pj;
        }
    }
}

/*
 * Usage:

    unsigned int* hist_gpu;
    float* host_sum_Pi;
    float* host_sum_i_Pi;
    float* host_u_0;
    float* host_varance;
    int* host_thres;

    cudaMalloc((void**)&host_sum_Pi,256* sizeof(float));
    cudaMalloc((void**)&host_sum_i_Pi,256* sizeof(float));
    cudaMalloc((void**)&host_u_0, sizeof(float));
    cudaMalloc((void**)&host_varance,256* sizeof(float));
    cudaMalloc((void**)&host_thres,sizeof(int));

 */

void ostu_gpu(unsigned char* dataIn,
              unsigned char* dataOut,
              unsigned int* hist,
              float* sum_Pi,
              float* sum_i_Pi,
              float* u_0,
              float* varance,
              int* thres,
              short int imgRows,
              short int imgCols)
{

    dim3 tPerBlock_hist(32,32);
    dim3 bPerGrid_hist((imgCols+32-1)/32,(imgRows+32-1)/32);

    getHist_gpu(dataIn,hist,tPerBlock_hist,bPerGrid_hist);

    dim3 tPerBlock_ostu(256,1);
    dim3 bPerGrid_ostu(257,1);
    ostu<<<bPerGrid_ostu,tPerBlock_ostu>>>(hist,sum_Pi,sum_i_Pi,u_0,varance,imgRows,imgCols);

    tree_max<<<1,256>>>(varance,thres);
    thresholdBinarization_inner<<<bPerGrid_hist,tPerBlock_hist>>>(dataIn,dataOut,imgRows,imgCols,thres);
}
