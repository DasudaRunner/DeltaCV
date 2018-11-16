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

    cv::Mat binarization_cuda_sobel(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));
    cv::Mat binarization_cuda_scharr(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));

    uchar3* frame_gpu;//读取的图像
    unsigned char* gray_gpu;//灰度图
    unsigned char* sobel_img_gpu;
    unsigned char* scharr_img_gpu;

    cudaMalloc((void**)&frame_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(uchar3));
    cudaMalloc((void**)&gray_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));
    cudaMalloc((void**)&sobel_img_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));
    cudaMalloc((void**)&scharr_img_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));

    dim3 threadsPerBlock(32,32);
    dim3 blockPerGrid((IMAGE_COLS+threadsPerBlock.x-1)/threadsPerBlock.x,(IMAGE_ROWS+threadsPerBlock.y-1)/threadsPerBlock.y);

    cv::Mat frame = cv::imread("/home/dasuda/david/cudaCV/imgs/usb_camera.jpg");

    while(true)
    {
        cudaMemcpy(frame_gpu,frame.data,IMAGE_ROWS*IMAGE_COLS * sizeof(uchar3),cudaMemcpyHostToDevice);

        RGB2GRAY_gpu(frame_gpu,gray_gpu,IMAGE_ROWS,IMAGE_COLS,threadsPerBlock,blockPerGrid);

        sobel_gpu(gray_gpu,sobel_img_gpu,IMAGE_ROWS,IMAGE_COLS,threadsPerBlock,blockPerGrid);
        cudaMemcpy(binarization_cuda_sobel.data,sobel_img_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

        scharr_gpu(gray_gpu,scharr_img_gpu,IMAGE_ROWS,IMAGE_COLS,threadsPerBlock,blockPerGrid);
        cudaMemcpy(binarization_cuda_scharr.data,scharr_img_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

        cv::imshow("frame",frame);
        cv::imshow("sobel",binarization_cuda_sobel);
        cv::imshow("scharr",binarization_cuda_scharr);

        if(cv::waitKey(3)>0)
        {
            cudaFree(frame_gpu);
            cudaFree(gray_gpu);
            cudaFree(sobel_img_gpu);
            cudaFree(scharr_img_gpu);
            break;
        }
    }


}
