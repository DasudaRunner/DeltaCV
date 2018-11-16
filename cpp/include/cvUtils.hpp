//
// Created by dasuda on 18-11-1.
//

#ifndef CUDACV_CVUTILS_HPP
#define CUDACV_CVUTILS_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;

void showHistImage(cv::Mat& dataIn, const unsigned int* hist, unsigned int bin)
{
    double maxVal = *max_element(hist,hist+bin-1);
    dataIn = cv::Mat(bin,bin,CV_8UC1,cv::Scalar(0));
    for (int i = 0; i < bin; ++i) {
        unsigned int val_hist = (hist[i]*bin)/maxVal;
        cv::line(dataIn,cv::Point(i,bin),cv::Point(i,bin-val_hist),cv::Scalar(255));
    }
}

class clocker {

    private:
        clock_t last_time;
        static const int slideLength = 500;
        double slideTimer[slideLength];
        int idSlide=0;
        int epoch=0;

    public:
        clocker()
        {
            memset(slideTimer,0.0, sizeof(double)*slideLength);
            idSlide=0;
        }

        void start()
        {
            last_time = clock();
        }

        void print_us(string ID)
        {
            cout<<"Total time["<<ID<<"]: "<<((double)((clock()-last_time)*1000000.0)/CLOCKS_PER_SEC)<<"us"<<endl;
        }

        void print_ms(string ID)
        {
            cout<<"Total time["<<ID<<"]: "<<((double)((clock()-last_time)*1000.0)/CLOCKS_PER_SEC)<<"ms"<<endl;
        }

        double getTime_ms()
        {
            return ((double)((clock()-last_time)*1000.0)/CLOCKS_PER_SEC);
        }

        double getTime_us()
        {
            return ((double)((clock()-last_time)*1000000.0)/CLOCKS_PER_SEC);
        }

        void print_ms_slideTimer(string ID,int div)
        {

            slideTimer[idSlide] = ((double)((clock()-last_time)*1000.0)/CLOCKS_PER_SEC);

            double sum_time = 0.0;
            for (int i = 0; i < slideLength; ++i) {
                sum_time += slideTimer[i];
            }

            float print_time = sum_time/slideLength/div;

            idSlide++;

            if(idSlide>=slideLength)
            {
                idSlide=0;
                epoch++;
                cout<<"SlideTimer ["<<ID<<"]["<<epoch<<"]: "<<print_time<<"ms"<<endl;
            }
        }
};

#endif //CUDACV_CVUTILS_HPP
