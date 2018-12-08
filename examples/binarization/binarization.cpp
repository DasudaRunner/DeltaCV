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
#include "deltaCV.hpp"

using namespace std;

#define IMAGE_ROWS 480
#define IMAGE_COLS 640

int main() {

    if(!getGPUConfig())
    {
        return 0;
    }

    cv::Mat dst;

    deltaCV::binarization binar(IMAGE_COLS,IMAGE_ROWS);
    deltaCV::colorSpace color(IMAGE_COLS,IMAGE_ROWS);

    cv::Mat frame = cv::imread("***.jpg");
    cv::Mat gray,opencv_ostu;

    while(true)
    {
        color.imgToGPU(frame);
        color.toGRAY();
        color.getMat(gray,0); //get cv::mat from gpu, taking a lot of time

        cv::threshold(gray, opencv_ostu, 0, 255, CV_THRESH_OTSU);

        binar.setGpuPtr(color.getGpuPtr_GRAY());
        binar.ostu();
        binar.getMat(dst);

        cv::imshow("frame",frame);
        cv::imshow("gray_cuda",dst);
        cv::imshow("opencv_ostu",opencv_ostu);

        if(cv::waitKey(3)>0)
        {
            break;
        }
    }


}