#include <iostream>
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>

#include "nodeCamera.hpp"

using namespace std;

int main()
{
    //托管共享内存
    boost::interprocess::managed_shared_memory managed_shm(boost::interprocess::open_only,SHARE_MEMORY_NAME);

    pair<int*,size_t > p = managed_shm.find<int>("integer");

    boost::interprocess::named_mutex named_mtx(boost::interprocess::open_only, LOCK_NAME);

    cv::Mat frame(IMAGE_HEIGHT,IMAGE_WIDTH,CV_8UC1,cv::Scalar(0));
    while(true)
    {
        named_mtx.lock();
        for (int i = 0; i < 480; ++i) {
            uchar* ptr_data = frame.ptr<uchar>(i);
            for (int j = 0; j < 640; ++j) {
                ptr_data[j] = *(p.first+i*640+j);
            }
        }
        named_mtx.unlock();
        cv::imshow("frame",frame);

        if(cv::waitKey(1)>0)
        {
            boost::interprocess::named_mutex::remove(LOCK_NAME);
            break;
        }
    }
}

