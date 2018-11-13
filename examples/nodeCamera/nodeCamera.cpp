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

#include <signal.h>

using namespace std;
cv::VideoCapture cap;
void signal_handler(int sig)
{
    cap.release();
    exit(0);
}

int main()
{
    signal(SIGINT,signal_handler);
    signal(SIGTSTP,signal_handler);

    bool res = boost::interprocess::shared_memory_object::remove(SHARE_MEMORY_NAME); //首先检查内存是否被释放
    boost::interprocess::named_mutex::remove(LOCK_NAME);

    cout<<"remove shm "<<res<<endl;

    //托管共享内存
    boost::interprocess::managed_shared_memory managed_shm(boost::interprocess::open_or_create,SHARE_MEMORY_NAME,2457601);

    int* shm_image = managed_shm.find_or_construct<int>("integer")[IMAGE_HEIGHT][IMAGE_WIDTH](0);

    boost::interprocess::named_mutex named_mtx(boost::interprocess::open_or_create, LOCK_NAME);

    cap.open(0);

    cv::Mat frame,gray;

    while(true)
    {
        cap>>frame;
        cv::cvtColor(frame,gray,cv::COLOR_RGB2GRAY);
        named_mtx.lock();
        for (int i = 0; i < IMAGE_HEIGHT; ++i) {
            uchar* ptr_data = gray.ptr<uchar>(i);
            for (int j = 0; j < IMAGE_WIDTH; ++j) {
                *(shm_image+i*IMAGE_WIDTH+j) = ptr_data[j];
            }
        }
        named_mtx.unlock();
    }
}
