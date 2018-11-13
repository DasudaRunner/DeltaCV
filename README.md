# DeltaCV
An open-source high performance library for image processing. Welcome to the new world.

&nbsp;[![author Haibo](https://img.shields.io/badge/author-Haibo%20Wong-blue.svg?style=flat)](https://github.com/DasudaRunner/Object-Tracking)&nbsp;&nbsp;&nbsp;&nbsp;
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)<br>
- &emsp;***Author**: Haibo Wang*<br>
- &emsp;***Email**: dasuda2015@163.com*
- &emsp;***Home Page**: <a href=dasuda.top>dasuda.top</a>*

## Performance Table

Image Size: 480*640（h*w）

| Function | GPU/ms (NVIDIA GTX 1070 8G) | CPU/ms (OpenCV) | Speed-up |
|:-:|:-:|:-:|:-:|
|RGB2GRAY|0.008 - 0.010|0.340 - 0.360|3.4 - 45|
|RGB2HSV|0.150 - 0.200|3.900 - 4.400|19.5 - 29.3|
|thresholdBinarization|0.005 - 0.008|0.035 - 0.045|4.4 - 9.0|
|sobel / scharr|-|0.032 - 0.038|-|
|erode / dilate (3*3 rect)|-|0.045 - 0.049|-|
|getHist (bin:256)|-|0.145 - 0.149|-|

## Function List

### Color space transformation

- **RGB2GRAY**`(uchar3* dataIn,unsigned char* dataOut,int imgRows,int imgCols)`: in `./cu/src/colorSpace.cu`. Converting RGB images to gray-scale images.

- **RGB2HSV**`(uchar3* dataIn,uchar3* dataOut,int imgRows,int imgCols,uchar3 minVal,uchar3 maxVal)`: in `./cu/src/colorSpace.cu`. Converting RGB images to HSV images， and using threshold segmentation to RGB images based on minVal and maxVal.

### Binarization

- **thresholdBinarization**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols,unsigned char thresholdMin,unsigned char thresholdMax,unsigned char valMin,unsigned char valMax)`: in `./cu/src/binarization.cu`. Similar to OpenCV, I designed 4 modes: `THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV`.

```cpp
/*
 * Compare 'threshold()' funciton in OpenCV
 * When:
 *      thresholdMin = thresholdMax+1 and valMin = 0  ==> THRESH_BINARY
 *      thresholdMin = thresholdMax+1 and valMax = 0  ==> THRESH_BINARY_INV
 *      thresholdMax = valMax and thresholdMin = 0  ==> THRESH_TRUNC
 *      thresholdMax = 255 and valMin = 0  ==> THRESH_TOZERO
 *      thresholdMin = 0 and valMax = 0  ==> THRESH_TOZERO_INV
 */
```

### Edge Detection

- **sobel**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols)`: in `./cu/src/edgeDetection.cu`. Edge detection using sobel operator.

- **scharr**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols)`: in `./cu/src/edgeDetection.cu`. Edge detection using scharr operator.

### Erode and dilate

- **erode**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols,short int erodeElementRows,short int erodeElementCols)`: in `./cu/src/erode_dilate.cu`. 

- **dilate**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols,short int dilateElementRows,short int dilateElementCols)`: in `./cu/src/erode_dilate.cu`. 

### Histogram

- **getHist**`(unsigned char* dataIn, unsigned int* hist)`： in `./cu/src/getHist.cu`. (**Unfinished**)
