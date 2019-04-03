#include "deltaCV/gpu/cudaImg.cuh"
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
 * 在原图上（RGB空间）直接利用HSV的各通道进行约束
 */

__global__ void RGB2HSV(uchar3* dataIn,
                          uchar3* dataOut,
                          int imgRows,
                          int imgCols,
                        uchar3 minVal,
                        uchar3 maxVal)
{
    int xIndex = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    int yIndex = threadIdx.y + __umul24(blockIdx.y, blockDim.y);

    int tid = yIndex*imgCols+xIndex;

    float maxPixel,minPixel;
    unsigned char sign0,sign1;
    float _h=0.0f,_s=0.0f;
    uchar3 zeroPixel;
    zeroPixel.x=0;
    zeroPixel.y=0;
    zeroPixel.z=0;

    if(xIndex < imgCols && yIndex < imgRows)
    {
        uchar3 rgb = dataIn[tid];

        unsigned char r = rgb.z;
        unsigned char g = rgb.y;
        unsigned char b = rgb.x;

        sign0 = max(r,g);
        maxPixel = max(sign0,b);

        sign1 = min(r,g);
        minPixel = min(sign1,b);

        float div = maxPixel-minPixel;

        if(maxPixel!=0)
        {
            _s = (div*255.0f)/maxPixel;
        }

        _h = (maxPixel==r)*((60*(g-b))/div+360)+
                (maxPixel==g)*((60*(b-r))/div+120)+
                (maxPixel==b)*((60*(r-g))/div+240);

        _h = (_h<0)*(_h + 360);

        _h /= 2.0f;

        if(_h >= minVal.x && _h <= maxVal.x &&
           _s >= minVal.y && _s <= maxVal.y &&
           minPixel >= minVal.z && minPixel <= maxVal.z)
        {
            dataOut[tid] = rgb;
        }else
        {
            dataOut[tid] = zeroPixel;
        }
    }
}

void RGB2HSV_gpu(uchar3* dataIn,
                 uchar3* dataOut,
                 int imgRows,
                 int imgCols,
                 uchar3 minVal,
                 uchar3 maxVal,
                   dim3 tPerBlock,
                   dim3 bPerGrid)
{
    RGB2HSV<<<bPerGrid,tPerBlock>>>(dataIn,dataOut,imgRows,imgCols,minVal,maxVal);
}
