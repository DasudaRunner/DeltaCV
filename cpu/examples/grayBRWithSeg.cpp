#include <iostream>
#include "deltaCV/SIMD/core.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <time.h>

using namespace std;
int main() {

    cv::Mat img = cv::imread("***.jpg");
    cv::Mat out1(img.rows,img.cols,CV_8UC1,cv::Scalar(0)),out2(img.rows,img.cols,CV_8UC1,cv::Scalar(0));
    cv::Mat s_colors[3],weighted_gray,normal_gray,opencvOUT,normal_out;

	deltaCV::grayBRWithSeg(img.data,out1.data,out2.data,img.cols,img.rows,
		               deltaCV::scalar_d(0.144,0.587,0.299),1,deltaCV::scalar(150),deltaCV::scalar(70));

	cv::split(img,s_colors);
	weighted_gray = s_colors[2]-s_colors[1];
	cv::threshold(weighted_gray,opencvOUT,70,255,cv::THRESH_BINARY);
	cv::cvtColor(img,normal_gray,cv::COLOR_BGR2GRAY);
	cv::threshold(normal_gray,normal_out,150,255,cv::THRESH_BINARY);

    cv::imshow("gray_scale",out1);
    cv::imshow("b/r-g",out2);

    cv::imshow("opencv",opencvOUT);
    cv::waitKey(0);

    return 0;
}
