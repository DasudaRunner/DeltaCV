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
#include "cvUtils.hpp"

using namespace std;

#define IMAGE_ROWS 480
#define IMAGE_COLS 640

int main() {

    if(!getGPUConfig())
    {
        return 0;
    }

    clocker timer;

    cv::Mat binarization_cuda_blur3(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));
    cv::Mat gray_cuda(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));

    uchar3* frame_gpu;//读取的图像
    unsigned char* gray_gpu;//灰度图
    unsigned char* blur3_gpu;//灰度图

    cudaMalloc((void**)&frame_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(uchar3));
    cudaMalloc((void**)&gray_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));
    cudaMalloc((void**)&blur3_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));

    dim3 threadsPerBlock(32,32);
    dim3 blockPerGrid((IMAGE_COLS+threadsPerBlock.x-1)/threadsPerBlock.x,(IMAGE_ROWS+threadsPerBlock.y-1)/threadsPerBlock.y);

    cv::Mat frame = cv::imread("/home/dasuda/david/cudaCV/imgs/usb_camera.jpg");

    while(true)
    {
        cudaMemcpy(frame_gpu,frame.data,IMAGE_ROWS*IMAGE_COLS * sizeof(uchar3),cudaMemcpyHostToDevice);

        RGB2GRAY_gpu(frame_gpu,gray_gpu,IMAGE_ROWS,IMAGE_COLS,threadsPerBlock,blockPerGrid);
        cudaMemcpy(gray_cuda.data,gray_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

        timer.start();
        guassianBlur3_gpu(gray_gpu,blur3_gpu,IMAGE_ROWS,IMAGE_COLS,threadsPerBlock,blockPerGrid);
        cudaDeviceSynchronize();
        timer.print_ms_slideTimer("01",1);
        cudaMemcpy(binarization_cuda_blur3.data,blur3_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

        cv::imshow("frame",frame);
        cv::imshow("gray_cuda",gray_cuda);
        cv::imshow("binarization_cuda_blur3",binarization_cuda_blur3);

        if(cv::waitKey(3)>0)
        {
            cudaFree(frame_gpu);
            cudaFree(gray_gpu);
            cudaFree(blur3_gpu);
            break;
        }
    }


}
