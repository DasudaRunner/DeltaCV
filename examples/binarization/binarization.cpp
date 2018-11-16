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

    /*
     * 最终显示的三幅图像
     */
    cv::Mat binarization_cuda_threshlod(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));
    cv::Mat binarization_cuda_ostu(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));
    cv::Mat binarization_opencv_ostu(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));

    cv::Mat gray_cuda(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));

    uchar3* frame_gpu;//读取的图像
    unsigned char* gray_gpu;//灰度图
    unsigned char* cuda_threshlod;
    unsigned char* cuda_ostu;

    /*
     * 中间变量，用于核函数之间的传递
     */
    unsigned int* hist_gpu;
    float* host_sum_Pi;
    float* host_sum_i_Pi;
    float* host_u_0;
    float* host_varance;
    int* host_thres;

    cudaMalloc((void**)&hist_gpu,256* sizeof(unsigned int));
    cudaMalloc((void**)&host_sum_Pi,256* sizeof(float));
    cudaMalloc((void**)&host_sum_i_Pi,256* sizeof(float));
    cudaMalloc((void**)&host_u_0, sizeof(float));
    cudaMalloc((void**)&host_varance,256* sizeof(float));
    cudaMalloc((void**)&host_thres,sizeof(int));

    cudaMalloc((void**)&frame_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(uchar3));
    cudaMalloc((void**)&gray_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));
    cudaMalloc((void**)&cuda_threshlod,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));
    cudaMalloc((void**)&cuda_ostu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));

    dim3 threadsPerBlock(32,32);
    dim3 blockPerGrid((IMAGE_COLS+threadsPerBlock.x-1)/threadsPerBlock.x,(IMAGE_ROWS+threadsPerBlock.y-1)/threadsPerBlock.y);

    cv::Mat frame = cv::imread("/home/dasuda/david/cudaCV/imgs/usb_camera.jpg");

    while(true)
    {
        cudaMemcpy(frame_gpu,frame.data,IMAGE_ROWS*IMAGE_COLS * sizeof(uchar3),cudaMemcpyHostToDevice);

        RGB2GRAY_gpu(frame_gpu,gray_gpu,IMAGE_ROWS,IMAGE_COLS,threadsPerBlock,blockPerGrid);
        cudaMemcpy(gray_cuda.data,gray_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

        thresholdBinarization_gpu(gray_gpu,cuda_threshlod,IMAGE_ROWS,IMAGE_COLS,180,200,0,255,threadsPerBlock,blockPerGrid);
        cudaMemcpy(binarization_cuda_threshlod.data,cuda_threshlod,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

        ostu_gpu(gray_gpu,cuda_ostu,hist_gpu,host_sum_Pi,host_sum_i_Pi,host_u_0,host_varance,host_thres,IMAGE_ROWS,IMAGE_COLS);
        cudaMemcpy(binarization_cuda_ostu.data,cuda_ostu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

        cv::threshold(gray_cuda, binarization_opencv_ostu, 0, 255, CV_THRESH_OTSU);

        cv::imshow("frame",frame);
        cv::imshow("gray_cuda",gray_cuda);
        cv::imshow("binarization_cuda_threshlod",binarization_cuda_threshlod);
        cv::imshow("binarization_cuda_ostu",binarization_cuda_ostu);
        cv::imshow("binarization_opencv_ostu",binarization_opencv_ostu);

        if(cv::waitKey(3)>0)
        {
            cudaFree(frame_gpu);
            cudaFree(gray_gpu);
            cudaFree(cuda_threshlod);
            cudaFree(cuda_ostu);

            cudaFree(hist_gpu);
            cudaFree(host_sum_Pi);
            cudaFree(host_sum_i_Pi);
            cudaFree(host_u_0);
            cudaFree(host_varance);
            cudaFree(host_thres);
            break;
        }
    }


}