#include <iostream>
#include "deltaCV/SIMD/core.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <time.h>

using namespace std;
int main() {

    cv::Mat img = cv::imread("***.jpg");
    cv::Mat out1(img.rows,img.cols,CV_8UC1,cv::Scalar(0));
    cv::Mat s_colors[3],weighted_gray,opencvOUT;
    int i=0;
    double deltaCVtime,openCVtime;//red 70


    clock_t stime=clock();
    deltaCV::grayBRWithSegStandard(img.data,out1.data,img.cols,img.rows,
                           0,deltaCV::scalar(150));

    cv::split(img,s_colors);
    weighted_gray = s_colors[2]-s_colors[0];
    cv::threshold(weighted_gray,opencvOUT,150,255,cv::THRESH_BINARY);

    cv::imshow("deltaCV",out1);
    cv::imshow("opencv",opencvOUT);
    cv::waitKey(0);

    return 0;
}
