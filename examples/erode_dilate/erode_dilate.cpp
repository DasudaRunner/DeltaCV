#include <iostream>
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#include "cudaImg.cuh"
#include "cudaUtils.hpp"
#include <time.h>
#include <algorithm>

using namespace std;

#define IMAGE_ROWS 480
#define IMAGE_COLS 640

int main() {

    if(!getGPUConfig())
    {
        return 0;
    }

    cv::Mat binarization_cuda_erode(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));
    cv::Mat binarization_cuda_dilate(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));

    uchar3* frame_gpu;//读取的图像
    unsigned char* gray_gpu;//灰度图
    unsigned char* erode_img_gpu;
    unsigned char* dilate_img_gpu;

    cudaMalloc((void**)&frame_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(uchar3));
    cudaMalloc((void**)&gray_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));
    cudaMalloc((void**)&erode_img_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));
    cudaMalloc((void**)&dilate_img_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));

    dim3 threadsPerBlock(32,32);
    dim3 blockPerGrid((IMAGE_COLS+threadsPerBlock.x-1)/threadsPerBlock.x,(IMAGE_ROWS+threadsPerBlock.y-1)/threadsPerBlock.y);

    cv::Mat frame = cv::imread("***.jpg");

    while(true)
    {
        cudaMemcpy(frame_gpu,frame.data,IMAGE_ROWS*IMAGE_COLS * sizeof(uchar3),cudaMemcpyHostToDevice);

        RGB2GRAY_gpu(frame_gpu,gray_gpu,IMAGE_ROWS,IMAGE_COLS,threadsPerBlock,blockPerGrid);

        erode_gpu(gray_gpu,erode_img_gpu,IMAGE_ROWS,IMAGE_COLS,cv::Size(5,5),threadsPerBlock,blockPerGrid);
        cudaMemcpy(binarization_cuda_erode.data,erode_img_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

        dilate_gpu(gray_gpu,dilate_img_gpu,IMAGE_ROWS,IMAGE_COLS,cv::Size(5,5),threadsPerBlock,blockPerGrid);
        cudaMemcpy(binarization_cuda_dilate.data,dilate_img_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

        cv::imshow("frame",frame);
        cv::imshow("erode",binarization_cuda_erode);
        cv::imshow("dilate",binarization_cuda_dilate);

        if(cv::waitKey(3)>0)
        {
            cudaFree(frame_gpu);
            cudaFree(gray_gpu);
            cudaFree(erode_img_gpu);
            cudaFree(dilate_img_gpu);
            break;
        }
    }


}
