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
    cv::Mat s_colors[3],weighted_gray,opencvOUT;
    double deltaCVtime,openCVtime;

	// deltaCV
	deltaCV::weightedGrayWithSeg(img.data,out.data,img.cols,img.rows,deltaCV::scalar_d(0.9,0.05,0.05),deltaCV::scalar(125),deltaCV::scalar(255));
	
	//OpenCV
	cv::split(img,s_colors);
	weighted_gray = s_colors[0]*0.9+s_colors[1]*0.05+s_colors[2]*0.05;
	cv::threshold(weighted_gray,opencvOUT,125,255,cv::THRESH_BINARY);

    cv::imshow("deltaCV",out);
    cv::imshow("opencv",opencvOUT);
    cv::waitKey(0);

    return 0;
}
