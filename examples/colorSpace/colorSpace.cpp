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

    cv::Mat binarization_cuda_hsv(IMAGE_ROWS,IMAGE_COLS,CV_8UC3,cv::Scalar(0));
    cv::Mat gray_cuda(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));

    uchar3* frame_gpu;//读取的图像
    uchar3* hsv_range;//经过hsv空间约束的图像
    unsigned char* gray_gpu;//灰度图

    cudaMalloc((void**)&frame_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(uchar3));
    cudaMalloc((void**)&hsv_range,IMAGE_ROWS*IMAGE_COLS* sizeof(uchar3));
    cudaMalloc((void**)&gray_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));

    dim3 threadsPerBlock(32,32);
    dim3 blockPerGrid((IMAGE_COLS+threadsPerBlock.x-1)/threadsPerBlock.x,(IMAGE_ROWS+threadsPerBlock.y-1)/threadsPerBlock.y);

    cv::Mat frame = cv::imread("/home/dasuda/david/cudaCV/imgs/usb_camera.jpg");

    while(true)
    {
        cudaMemcpy(frame_gpu,frame.data,IMAGE_ROWS*IMAGE_COLS * sizeof(uchar3),cudaMemcpyHostToDevice);

        RGB2GRAY_gpu(frame_gpu,gray_gpu,IMAGE_ROWS,IMAGE_COLS,threadsPerBlock,blockPerGrid);
        cudaMemcpy(gray_cuda.data,gray_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

        //这里我们需要使用uchar3类型的变量来表示范围
        uchar3 min_hsv={0,0,100},max_hsv={180,255,200};
        RGB2HSV_gpu(frame_gpu,hsv_range,IMAGE_ROWS,IMAGE_COLS,min_hsv,max_hsv,threadsPerBlock,blockPerGrid);
        cudaMemcpy(binarization_cuda_hsv.data,hsv_range, IMAGE_ROWS * IMAGE_COLS * sizeof(uchar3),cudaMemcpyDeviceToHost);

        cv::imshow("frame",frame);
        cv::imshow("gray_cuda",gray_cuda);
        cv::imshow("binarization_cuda_hsv",binarization_cuda_hsv);

        if(cv::waitKey(3)>0)
        {
            cudaFree(frame_gpu);
            cudaFree(gray_gpu);
            cudaFree(hsv_range);
            break;
        }
    }


}