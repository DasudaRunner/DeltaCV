#include <iostream>
#include "deltaCV/SIMD/core.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <time.h>
using namespace std;
int main() {

    cv::Mat img = cv::imread("***.bmp");
    cv::Mat out(img.rows,img.cols,CV_8UC1,cv::Scalar(0));
    cv::Mat opencvOUT;
    while(1)
    {

        deltaCV::inRange(img.data,out.data,img.cols,img.rows,deltaCV::scalar(100,0,0),deltaCV::scalar(255,255,255));

        cv::inRange(img,cv::Scalar(100,0,0),cv::Scalar(255,255,255),opencvOUT);

        cv::imshow("src",img);
        cv::imshow("out",out);
        cv::imshow("cvout",opencvOUT);
        cv::waitKey(1);
    }

    return 0;
}
