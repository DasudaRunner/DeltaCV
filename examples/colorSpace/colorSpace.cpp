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

    cv::Mat gray_img;
    cv::Mat hsv_img;

    cv::Mat frame = cv::imread("***.jpg");

    deltaCV::colorSpace color(IMAGE_COLS,IMAGE_ROWS);

    while(true)
    {

        color.imgToGPU(frame);

        uchar3 min_hsv={0,0,0},max_hsv={180,255,255};
        color.HSV_Segmentation(min_hsv,max_hsv);

        color.setGpuPtr(color.getGpuPtr_RGB());
        color.toGRAY();

        color.getMat(gray_img,0); //get gray image

        cv::imshow("frame",frame);
        cv::imshow("gray_cuda",gray_img);

        if(cv::waitKey(3)>0)
        {
            break;
        }
    }


}