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
    cv::Mat histGray(IMAGE_ROWS,IMAGE_COLS,CV_8UC1,cv::Scalar(0));

    unsigned int hist_cpu[256];
    memset(hist_cpu,0,256* sizeof(unsigned int));

    uchar3* d_in;
    unsigned char* gray_gpu;
    uchar3* hsvRange_gpu;
    unsigned char* erode_img_gpu;

    unsigned int* hist_gpu;

    unsigned int* sum_ni;
    unsigned char* histGray_gpu;

    cudaMalloc((void**)&sum_ni,256* sizeof(unsigned int));
    cudaMalloc((void**)&histGray_gpu,IMAGE_ROWS*IMAGE_COLS*sizeof(unsigned char));

    cudaMalloc((void**)&d_in,IMAGE_ROWS*IMAGE_COLS* sizeof(uchar3));
    cudaMalloc((void**)&gray_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));

    cudaMalloc((void**)&erode_img_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(unsigned char));
    cudaMalloc((void**)&hsvRange_gpu,IMAGE_ROWS*IMAGE_COLS* sizeof(uchar3));

    cudaMalloc((void**)&hist_gpu,256* sizeof(unsigned int));

    dim3 threadsPerBlock(32,32);
    dim3 blockPerGrid((IMAGE_COLS+threadsPerBlock.x-1)/threadsPerBlock.x,(IMAGE_ROWS+threadsPerBlock.y-1)/threadsPerBlock.y);

//    cv::VideoCapture cap0(0);

    cv::Mat frame = cv::imread("/home/dasuda/david/cudaCV/imgs/img0.jpg");
    cv::Mat bina_test;

    cv::Mat frame0,frame1;
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
        cudaDeviceSynchronize();
//        myTimer.start();
//        getHist_gpu(gray_gpu,hist_gpu,threadsPerBlock,blockPerGrid);
//        cudaMemcpy(hist_cpu,hist_gpu,256 * sizeof(unsigned int),cudaMemcpyDeviceToHost);
//        showHistImage(histImg,hist_cpu,256);
        equalizeHist_gpu(gray_gpu,hist_gpu,sum_ni,histGray_gpu,IMAGE_ROWS,IMAGE_COLS,threadsPerBlock,blockPerGrid);
        cudaDeviceSynchronize();
//        myTimer.print_ms_slideTimer("01",1);
//        cudaMemcpy(hist_cpu,sum_ni,256 * sizeof(unsigned int),cudaMemcpyDeviceToHost);
//        cout<<hist_cpu[0]<<endl;
//        cout<<hist_cpu[1]<<endl;
//        cout<<"----------"<<endl;
//        showHistImage(histImg,hist_cpu,256);
        cudaMemcpy(histGray.data,histGray_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);

//        ostu_gpu(gray_gpu,erode_img_gpu,hist_gpu,host_sum_Pi,host_sum_i_Pi,host_u_0,host_varance,host_thres,IMAGE_ROWS,IMAGE_COLS);

//        cudaMemcpy(erode_img.data,erode_img_gpu,IMAGE_ROWS*IMAGE_COLS * sizeof(unsigned char),cudaMemcpyDeviceToHost);
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

        myTimer.start();
        cv::equalizeHist(gray, bina_test);
        cv::equalizeHist(gray, bina_test);
        cv::equalizeHist(gray, bina_test);
        cv::equalizeHist(gray, bina_test);
        cv::equalizeHist(gray, bina_test);
        myTimer.print_ms_slideTimer("01",5);


        cv::imshow("frame",histGray);
        cv::imshow("bina_test",bina_test);
//        cv::imshow("gray",gray);
//        cv::imshow("hsvRange",hsvRange_img);
//        cv::imshow("erode",erode_img);
        
        if(cv::waitKey(3)>0)
        {
//            cv::imwrite("/home/dasuda/david/cudaCV/imgs/hist00.jpg",histImg);
//            boost::interprocess::named_mutex::remove("mtx");
            cudaFree(d_in);
            cudaFree(gray_gpu);
            cudaFree(erode_img_gpu);
            break;
        }
    }


}