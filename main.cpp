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

#include <cvUtils.hpp>

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

    clocker myTimer;

    cv::Mat gray(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));
    cv::Mat erode_img(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));
    cv::Mat hsvRange_img(IMAGE_ROWS,IMAGE_COLS,CV_8UC3,cv::Scalar(0));
    cv::Mat histImg;

    unsigned int hist_cpu[256];
    memset(hist_cpu,0,256* sizeof(int));

    uchar3* d_in;
    unsigned char* gray_gpu;
    uchar3* hsvRange_gpu;
    unsigned char* erode_img_gpu;

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

    cudaMalloc((void**)&d_in,IMAGE_ROWS*IMAGE_COLS* sizeof(uchar3));
    cudaMalloc((void**)&gray_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));

    cudaMalloc((void**)&erode_img_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));
    cudaMalloc((void**)&hsvRange_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(uchar3));

    cudaMalloc((void**)&hist_gpu,256* sizeof(unsigned int));

    dim3 threadsPerBlock(32,32);
    dim3 blockPerGrid((IMAGE_COLS+threadsPerBlock.x-1)/threadsPerBlock.x,(IMAGE_ROWS+threadsPerBlock.y-1)/threadsPerBlock.y);

    cv::VideoCapture cap(0);

    cv::Mat frame = cv::imread("/home/dasuda/david/cudaCV/imgs/usb_camera.jpg");
    cv::Mat bina_test;
    while(true)
    {
        //cap>>frame;

        cudaMemcpy(d_in,frame.data,IMAGE_ROWS*IMAGE_COLS * sizeof(uchar3),cudaMemcpyHostToDevice);

//        uchar3 min_hsv={0,0,0},max_hsv={180,255,255};
//        RGB2HSV_gpu(d_in,hsvRange_gpu,IMAGE_ROWS,IMAGE_COLS,min_hsv,max_hsv,threadsPerBlock,blockPerGrid);
//        cudaMemcpy(hsvRange_img.data,hsvRange_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(uchar3),cudaMemcpyDeviceToHost);

//        myTimer.start();
        RGB2GRAY_gpu(d_in,gray_gpu,IMAGE_ROWS,IMAGE_COLS,threadsPerBlock,blockPerGrid);
//        cudaDeviceSynchronize();
//        myTimer.print_ms_slideTimer("01");
        cudaMemcpy(gray.data,gray_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

//        thresholdBinarization_gpu(gray_gpu,erode_img_gpu,IMAGE_ROWS,IMAGE_COLS,180,200,0,255,threadsPerBlock,blockPerGrid);
//        cudaMemcpy(erode_img.data,erode_img_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

//        sobel_gpu(gray_gpu,erode_img_gpu,IMAGE_ROWS,IMAGE_COLS,threadsPerBlock,blockPerGrid);
//        erode_gpu(gray_gpu,erode_img_gpu,IMAGE_ROWS,IMAGE_COLS,cv::Size(3,3),threadsPerBlock,blockPerGrid);

//        getHist_gpu(gray_gpu,hist_gpu,threadsPerBlock,blockPerGrid);
//        cudaMemcpy(hist_cpu,hist_gpu,256 * sizeof(unsigned int),cudaMemcpyDeviceToHost);
//        showHistImage(histImg,hist_cpu,256);
        cudaDeviceSynchronize();
        myTimer.start();
        ostu_gpu(gray_gpu,erode_img_gpu,hist_gpu,host_sum_Pi,host_sum_i_Pi,host_u_0,host_varance,host_thres,IMAGE_ROWS,IMAGE_COLS);
        cudaDeviceSynchronize();
        myTimer.print_ms_slideTimer("01",1);

        cudaMemcpy(erode_img.data,erode_img_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);
//        cudaMemcpy(hist_cpu,hist_gpu,256 * sizeof(unsigned int),cudaMemcpyDeviceToHost);
//        showHistImage(histImg,hist_cpu,256);

//        cudaMemcpy(erode_img.data,erode_img_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

//        myTimer.start();
//        cv::threshold(gray,bina_test,180,255,cv::THRESH_BINARY);
//        cv::threshold(gray,bina_test,180,255,cv::THRESH_BINARY);
//        cv::threshold(gray,bina_test,180,255,cv::THRESH_BINARY);
//        cv::threshold(gray,bina_test,180,255,cv::THRESH_BINARY);
//        cv::threshold(gray,bina_test,180,255,cv::THRESH_BINARY);
//
//        myTimer.print_ms_slideTimer("01");

//        thresholdBinarization_gpu(gray_gpu,erode_img_gpu,IMAGE_ROWS,IMAGE_COLS,180,200,0,255,threadsPerBlock,blockPerGrid);
//        cudaMemcpy(erode_img.data,erode_img_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

//        showHistImage(histImg,hist_cpu,256);

//        cv::cvtColor(frame,gray,cv::COLOR_RGB2GRAY);
//        named_mtx.lock();
//        for (int i = 0; i < 480; ++i) {
//            uchar* ptr_data = gray.ptr<uchar>(i);
//            for (int j = 0; j < 640; ++j) {
//                *(shm_image+i*640+j) = ptr_data[j];
//            }
//        }
//        named_mtx.unlock();

//        myTimer.start();
        cv::threshold(gray, bina_test, 0, 255, CV_THRESH_OTSU);
//        myTimer.print_ms_slideTimer("01",5);
//        cv::imshow("frame",histImg);
        cv::imshow("gray",gray);
//        cv::imshow("hsvRange",hsvRange_img);
        cv::imshow("erode",erode_img);
        cv::imshow("opencv_threshold",bina_test);

        if(cv::waitKey(3)>0)
        {
//            cv::imwrite("/home/dasuda/david/cudaCV/imgs/usb_camera.jpg",frame);
//            boost::interprocess::named_mutex::remove("mtx");
            cudaFree(d_in);
            cudaFree(gray_gpu);
            cudaFree(erode_img_gpu);
            break;
        }
    }


}