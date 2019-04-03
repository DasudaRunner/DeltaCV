#include <iostream>
#include "deltaCV/SIMD/core.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <time.h>
using namespace std;
int main() {

    cv::Mat img = cv::imread("/home/dasuda/david/SSE/img0.bmp");
    cv::Mat out(img.rows,img.cols,CV_8UC1,cv::Scalar(0));
    cv::Mat opencvOUT;
    int i=0;
    double deltaCVtime,openCVtime;
    while(i<200)
    {
        i++;
        clock_t stime=clock();
        deltaCV::inRange(img.data,out.data,img.cols,img.rows,deltaCV::scalar(100,0,0),deltaCV::scalar(255,255,255));
        deltaCV::inRange(img.data,out.data,img.cols,img.rows,deltaCV::scalar(100,0,0),deltaCV::scalar(255,255,255));
        deltaCV::inRange(img.data,out.data,img.cols,img.rows,deltaCV::scalar(100,0,0),deltaCV::scalar(255,255,255));
        deltaCV::inRange(img.data,out.data,img.cols,img.rows,deltaCV::scalar(100,0,0),deltaCV::scalar(255,255,255));
        deltaCV::inRange(img.data,out.data,img.cols,img.rows,deltaCV::scalar(100,0,0),deltaCV::scalar(255,255,255));
        deltaCVtime += (double)(clock()-stime)/CLOCKS_PER_SEC*1000.0;

        stime=clock();
        cv::inRange(img,cv::Scalar(100,0,0),cv::Scalar(255,255,255),opencvOUT);
        cv::inRange(img,cv::Scalar(100,0,0),cv::Scalar(255,255,255),opencvOUT);
        cv::inRange(img,cv::Scalar(100,0,0),cv::Scalar(255,255,255),opencvOUT);
        cv::inRange(img,cv::Scalar(100,0,0),cv::Scalar(255,255,255),opencvOUT);
        cv::inRange(img,cv::Scalar(100,0,0),cv::Scalar(255,255,255),opencvOUT);
        openCVtime += (double)(clock()-stime)/CLOCKS_PER_SEC*1000.0;

//        cv::imshow("src",img);
//        cv::imshow("out",out);
//        cv::imshow("cvout",opencvOUT);
//        cv::waitKey(1);
    }
    cout<<"deltaCV: "<<deltaCVtime/(5*i)<<endl;
    cout<<"openCV: "<<openCVtime/(5*i)<<endl;

    return 0;
}
