//
// Created by dasuda on 18-10-27.
//

#ifndef CUDAIMG_CUH
#define CUDAIMG_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

extern "C"{

    //***************colorSpace***************
    void RGB2GRAY_gpu(uchar3* dataIn,
                      unsigned char* dataOut,
                      int imgRows,
                      int imgCols,
                      dim3 tPerBlock,
                      dim3 bPerGrid);

    void RGB2HSV_gpu(uchar3* dataIn,
                     uchar3* dataOut,
                     int imgRows,
                     int imgCols,
                     uchar3 minVal,
                     uchar3 maxVal,
                     dim3 tPerBlock,
                     dim3 bPerGrid);

    //***************getHist***************
    void getHist_gpu(unsigned char* dataIn,
                     unsigned int* hist,
                     dim3 tPerBlock,
                     dim3 bPerGrid);

    //***************erode_dilate***************
    void erode_gpu(unsigned char* dataIn,
                   unsigned char* dataOut,
                    short int imgRows,
                    short int imgCols,
                   cv::Size erodeSize,
                   dim3 tPerBlock,
                   dim3 bPerGrid);

    void dilate_gpu( unsigned char* dataIn,
                     unsigned char* dataOut,
                     short int imgRows,
                     short int imgCols,
                    cv::Size dilateSize,
                    dim3 tPerBlock,
                    dim3 bPerGrid);

    //***************edgeDetection***************
    void sobel_gpu( unsigned char* dataIn,
                    unsigned char* dataOut,
                    short int imgRows,
                    short int imgCols,
                   dim3 tPerBlock,
                   dim3 bPerGrid);

    void scharr_gpu( unsigned char* dataIn,
                     unsigned char* dataOut,
                    short int imgRows,
                    short int imgCols,
                   dim3 tPerBlock,
                   dim3 bPerGrid);

    //***************binarization***************
    void thresholdBinarization_gpu(unsigned char* dataIn,
                                   unsigned char* dataOut,
                                   short int imgRows,
                                   short int imgCols,
                                   unsigned char thresholdMin,
                                   unsigned char thresholdMax,
                                   unsigned char valMin,
                                   unsigned char valMax,
                                   dim3 tPerBlock,
                                   dim3 bPerGrid);

    void ostu_gpu(unsigned char* dataIn,
                  unsigned char* dataOut,
                  unsigned int* hist,
                  float* sum_Pi,
                  float* sum_i_Pi,
                  float* u_0,
                  float* varance,
                  int* thres,
                  short int imgRows,
                  short int imgCols);

    //***************blur***************
    void guassianBlur3_gpu(unsigned char* dataIn,
                          unsigned char* dataOut,
                          short int imgRows,
                          short int imgCols,
                          dim3 tPerBlock,
                          dim3 bPerGrid);
};

#endif //CUDAIMG_CUH
