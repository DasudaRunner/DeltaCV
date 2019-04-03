# DeltaCV
An open-source high performance library for image processing. including CPU optimization and GPU optimization. PRs are welcome.

&nbsp;[![author Haibo](https://img.shields.io/badge/author-Haibo%20Wong-blue.svg?style=flat)](https://github.com/DasudaRunner/Object-Tracking)&nbsp;&nbsp;&nbsp;&nbsp;
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)<br>
- &emsp;***Author**: Haibo Wang*<br>
- &emsp;***Email**: dasuda2015@163.com*
- &emsp;***Home Page**: [dasuda.top](https://dasuda.top)*

---
## 1. Samples

All samples are in `examples/`. 

- [x] **binarization**
- [ ] **colorSpace**
- [ ] **edgeDetection**
- [ ] **erode_dilate**
- [ ] **getHist**
- [ ] **equalizeHist**
- [ ] **blur**


## 2. Shared Memory

### Dependencies

- Boost

### Location

`cpu/include/deltaCV/cpu/shm.hpp`

### Include
```cpp
#include "deltaCV/cpu/shm.hpp"
```
For more details， see [my blog](https://dasuda.top/deltacv/2019/04/02/DeltaCV%E4%B9%8B%E5%85%B1%E4%BA%AB%E5%86%85%E5%AD%98%E7%AF%87/);

---
## 3. CUDA

### Dependencies

- CUDA
- OpenCV

### Performance Table

Image Size: 480 x 640（H x W）

| Function | GPU/ms (NVIDIA GTX 1070 8G) | CPU/ms (OpenCV on i5 7500) | Speed-up |
|:-:|:-:|:-:|:-:|
|RGB2GRAY|0.008 - 0.010|0.340 - 0.360|3.4 - 45|
|RGB2HSV|0.150 - 0.200|3.900 - 4.400|19.5 - 29.3|
|thresholdBinarization|0.005 - 0.008|0.035 - 0.045|4.4 - 9.0|
|ostu|0.16-0.17|1.280-1.432|8.0-8.9|
|sobel / scharr|0.032 - 0.038|-|-|
|erode / dilate (3*3 rect)|0.045 - 0.049|-|-|
|getHist (bin:256)|0.145 - 0.149|-|-|
|equalizeHist(bin:256)|0.16-0.17|0.31-0.32|1.8-2.0|
|blur(3*3 guassian kernel)|0.036-0.040|-|-|

### Function List

#### Color space transformation

- **RGB2GRAY**`(uchar3* dataIn,unsigned char* dataOut,int imgRows,int imgCols)`: in `gpu/src/colorSpace.cu`. Converting RGB images to gray-scale images.

- **RGB2HSV**`(uchar3* dataIn,uchar3* dataOut,int imgRows,int imgCols,uchar3 minVal,uchar3 maxVal)`: in `gpu/src/colorSpace.cu`. Converting RGB images to HSV images， and using threshold segmentation to RGB images based on minVal and maxVal.

#### Binarization

- **thresholdBinarization**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols,unsigned char thresholdMin,unsigned char thresholdMax,unsigned char valMin,unsigned char valMax)`: in `gpu/src/binarization.cu`. Similar to OpenCV function `threshold()`, I designed 5 modes: `THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV`.

```cpp
/*
 * Compare 'threshold()' funciton in OpenCV
 * When:
 *      thresholdMin = thresholdMax and valMin = 0  ==> THRESH_BINARY
 *      thresholdMin = thresholdMax and valMax = 0  ==> THRESH_BINARY_INV
 *      thresholdMax = valMax and thresholdMin = 0  ==> THRESH_TRUNC
 *      thresholdMax = 255 and valMin = 0  ==> THRESH_TOZERO
 *      thresholdMin = 0 and valMax = 0  ==> THRESH_TOZERO_INV
 */
```

- **ostu_gpu**`(unsigned char* dataIn,unsigned char* dataOut,unsigned int* hist,float* sum_Pi,float* sum_i_Pi,float* u_0,float* varance,int* thres,short int imgRows,short int imgCols)`: in `gpu/src/binarization.cu`. Binarization using ostu.

#### Edge Detection

- **sobel**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols)`: in `gpu/src/edgeDetection.cu`. Edge detection using sobel operator.

- **scharr**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols)`: in `gpu/src/edgeDetection.cu`. Edge detection using scharr operator.

#### Erode and Dilate

- **erode**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols,short int erodeElementRows,short int erodeElementCols)`: in `gpu/src/erode_dilate.cu`. 

- **dilate**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols,short int dilateElementRows,short int dilateElementCols)`: in `gpu/src/erode_dilate.cu`. 

#### Histogram

- **getHist**`(unsigned char* dataIn, unsigned int* hist)`： in `gpu/src/getHist.cu`. 
- **\[wrapper\]equalizeHist_gpu**`(unsigned char* dataIn,unsigned int* hist,unsigned int* sum_ni,unsigned char* dataOut,short int imgRows,short int imgCols,dim3 tPerBlock,dim3 bPerGrid)`： in `gpu/src/getHist.cu`.(**Unfinished**)

#### Guassian Blur

- **guassianBlur3_gpu**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols,dim3 tPerBlock,dim3 bPerGrid)`: in `gpu/src/blur.cu`. Guassian blur with 3\*3 kernel
