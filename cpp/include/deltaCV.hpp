#ifndef CUDACV_DELTACV_HPP
#define CUDACV_DELTACV_HPP

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#include "cudaImg.cuh"
#include "cudaUtils.hpp"

namespace deltaCV{

    using namespace std;

    class colorSpace {

    public:
        /** @brief

        @param img_cols COLS of image.
        @param img_rows ROWS of image.
        */
        colorSpace(int img_cols,int img_rows);
        ~colorSpace();

        /** @brief put cv::mat type image on GPU. Do not use it frequently.
            This function will result in considerable delays.

        @param src Input image.
        */
        void imgToGPU(cv::Mat& src);

        /** @brief set gpu ptr to this->frame_gpu.

        @param rgb_ptr Gpu ptr.
        */
        void setGpuPtr(uchar3* rgb_ptr);

        /** @brief convert image from rgb to gray.

        @param None.
        */
        void toGRAY();

        /** @brief Segmentation in image HSV color space.

        @param min_val Lower boundary with uchar3 type.
        @param max_val Upper boundary with uchar3 type.
        */
        void HSV_Segmentation(uchar3 min_val,uchar3 max_val);

        /** @brief Get the output from GPU.

        @param out Output array of the same size as src and CV_8U type.
        @param type If 0, return the gray image. if 1, return the rgb image.
        */
        void getMat(cv::Mat &out,int type);

        /** @brief get gray image gpu ptr.

        @param None.
        */
        unsigned char* getGpuPtr_GRAY();

        /** @brief get rgb image gpu ptr.

        @param None.
        */
        uchar3* getGpuPtr_RGB();

    private:
        int img_w,img_h;
        dim3 threadsPerBlock;
        dim3 blockPerGrid;

        uchar3* frame_gpu;
        unsigned char* gray_gpu;
        uchar3* hsv_gpu;

    };

    colorSpace::colorSpace(int img_cols,int img_rows) {
        this->img_w = img_cols;
        this->img_h = img_rows;

        threadsPerBlock = dim3(32,32);
        blockPerGrid = dim3(this->img_w + this->threadsPerBlock.x - 1,this->img_h + this->threadsPerBlock.y - 1);

        cudaMalloc((void**)&this->frame_gpu,this->img_h*this->img_w * sizeof(uchar3));
        cudaMalloc((void**)&this->gray_gpu,this->img_h*this->img_w * sizeof(unsigned char));
        cudaMalloc((void**)&this->hsv_gpu,this->img_h*this->img_w * sizeof(uchar3));
    }

    colorSpace::~colorSpace() {
        cudaFree(this->frame_gpu);
        cudaFree(this->gray_gpu);
        cudaFree(this->hsv_gpu);
    }

    void colorSpace::imgToGPU(cv::Mat &src) {
        cudaMemcpy(this->frame_gpu,src.data,this->img_h*this->img_w * sizeof(uchar3),cudaMemcpyHostToDevice);
    }

    void colorSpace::setGpuPtr(uchar3* rgb_ptr) {
        this->frame_gpu = rgb_ptr;
    }

    void colorSpace::getMat(cv::Mat &out,int type) {
        assert(type<=1);

        if(type==0)
        {
            out = cv::Mat(this->img_h,this->img_w,CV_8UC1,cv::Scalar(0));
            cudaMemcpy(out.data,this->gray_gpu,this->img_h*this->img_w * sizeof(unsigned char),cudaMemcpyDeviceToHost);
        }
        else
        {
            out = cv::Mat(this->img_h,this->img_w,CV_8UC3,cv::Scalar(0));
            cudaMemcpy(out.data,this->hsv_gpu, this->img_h*this->img_w * sizeof(uchar3),cudaMemcpyDeviceToHost);
        }
    }

    unsigned char* colorSpace::getGpuPtr_GRAY(){
        return this->gray_gpu;
    }

    uchar3* colorSpace::getGpuPtr_RGB() {
        return this->hsv_gpu;
    }

    void colorSpace::toGRAY() {
        RGB2GRAY_gpu(this->frame_gpu,this->gray_gpu,this->img_h,this->img_w,dim3(32,32),dim3(this->img_w+32-1,this->img_h+32-1));
    }

    void colorSpace::HSV_Segmentation(uchar3 min_val,uchar3 max_val) {
        RGB2HSV_gpu(this->frame_gpu,this->hsv_gpu,this->img_h,this->img_w,min_val,max_val,dim3(32,32),dim3(this->img_w+32-1,this->img_h+32-1));
    }
}

namespace deltaCV{
    using namespace std;

    class binarization{
    public:
        /** @brief

        @param img_cols COLS of image.
        @param img_rows ROWS of image.
        */
        binarization(int img_cols,int img_rows);
        ~binarization();

        /** @brief put cv::mat type image on GPU. Do not use it frequently.
            This function will result in considerable delays.

        @param src Input image.
        */
        void imgToGPU(cv::Mat& src);

        /** @brief set gpu ptr.

        @param ptr Gpu ptr.
        */
        void setGpuPtr(unsigned char* ptr);

        /** @brief threshold binarization.
         *
            Compare with 'threshold()' funciton in OpenCV

                thresholdMin = thresholdMax and valMin = 0  ==> THRESH_BINARY
                thresholdMin = thresholdMax and valMax = 0  ==> THRESH_BINARY_INV
                thresholdMax = valMax and thresholdMin = 0  ==> THRESH_TRUNC
                thresholdMax = 255 and valMin = 0  ==> THRESH_TOZERO
                thresholdMin = 0 and valMax = 0  ==> THRESH_TOZERO_INV

        @param thresholdMin Lower boundary.
        @param thresholdMax upper boundary.
        @param valMin Lower value.
        @param valMax upper value.
        */
        void thresholdBinarization(unsigned char thresholdMin,
                                   unsigned char thresholdMax,
                                   unsigned char valMin,
                                   unsigned char valMax);

        /** @brief ostu binarization.

        @param None.
        */
        void ostu();

        /** @brief Get the output from GPU.

        @param out Output array of the same size as src and CV_8U type.
        */
        void getMat(cv::Mat& out);

        /** @brief get gpu ptr.

        @param None.
        */
        unsigned char* getGpuPtr();
    private:
        int img_w,img_h;
        dim3 threadsPerBlock;
        dim3 blockPerGrid;

        unsigned char* frame_gpu;
        unsigned char* bin_gpu;

        //ostu
        unsigned int* hist_gpu;
        float* host_sum_Pi;
        float* host_sum_i_Pi;
        float* host_u_0;
        float* host_varance;
        int* host_thres;

    };
    binarization::binarization(int img_cols,int img_rows){
        this->img_w = img_cols;
        this->img_h = img_rows;

        threadsPerBlock = dim3(32,32);
        blockPerGrid = dim3(this->img_w + this->threadsPerBlock.x - 1,this->img_h + this->threadsPerBlock.y - 1);

        cudaMalloc((void**)&hist_gpu,256* sizeof(unsigned int));
        cudaMalloc((void**)&host_sum_Pi,256* sizeof(float));
        cudaMalloc((void**)&host_sum_i_Pi,256* sizeof(float));
        cudaMalloc((void**)&host_u_0, sizeof(float));
        cudaMalloc((void**)&host_varance,256* sizeof(float));
        cudaMalloc((void**)&host_thres,sizeof(int));

        cudaMalloc((void**)&frame_gpu,this->img_h*this->img_w* sizeof(unsigned char));
        cudaMalloc((void**)&bin_gpu,this->img_h*this->img_w* sizeof(unsigned char));

    }
    binarization::~binarization() {
        cudaFree(frame_gpu);
        cudaFree(bin_gpu);

        cudaFree(hist_gpu);
        cudaFree(host_sum_Pi);
        cudaFree(host_sum_i_Pi);
        cudaFree(host_u_0);
        cudaFree(host_varance);
        cudaFree(host_thres);
    }
    void binarization::imgToGPU(cv::Mat &src) {
        cudaMemcpy(this->frame_gpu,src.data,this->img_h*this->img_w * sizeof(unsigned char),cudaMemcpyHostToDevice);
    }

    void binarization::setGpuPtr(unsigned char *ptr) {
        this->frame_gpu = ptr;
    }

    void binarization::thresholdBinarization(unsigned char thresholdMin,
                                             unsigned char thresholdMax,
                                             unsigned char valMin,
                                             unsigned char valMax) {
        thresholdBinarization_gpu(this->frame_gpu,
                                  this->bin_gpu,
                                  this->img_h,this->img_w,
                                  thresholdMin,thresholdMax,valMin,valMax,
                                  this->threadsPerBlock,this->blockPerGrid);
    }

    void binarization::ostu() {
        ostu_gpu(this->frame_gpu,
                 this->bin_gpu,
                 this->hist_gpu,this->host_sum_Pi,this->host_sum_i_Pi,this->host_u_0,this->host_varance,this->host_thres,
                 this->img_h,this->img_w);
    }

    void binarization::getMat(cv::Mat &out) {
        out = cv::Mat(this->img_h,this->img_w,CV_8UC1,cv::Scalar(0));
        cudaMemcpy(out.data,this->bin_gpu,this->img_h*this->img_w * sizeof(unsigned char),cudaMemcpyDeviceToHost);
    }

    unsigned char* binarization::getGpuPtr() {
        return this->bin_gpu;
    }
}



#endif //CUDACV_DELTACV_HPP
