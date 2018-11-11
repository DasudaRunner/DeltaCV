# CUDA-CV
An open-source cuda library for image processing

&nbsp;[![author Haibo](https://img.shields.io/badge/author-Haibo%20Wong-blue.svg?style=flat)](https://github.com/DasudaRunner/Object-Tracking)&nbsp;&nbsp;&nbsp;&nbsp;
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)<br>

## Function List

### Color space transformation

- **RGB2GRAY**`(uchar3* dataIn,unsigned char* dataOut,int imgRows,int imgCols)`: in `./cu/src/colorSpace.cu`. Converting RGB images to gray-scale images.

- **RGB2HSV_V**`(uchar3* dataIn,uchar3* dataOut,int imgRows,int imgCols,short int minVal,short int maxVal)`: in `./cu/src/colorSpace.cu`. Converting RGB images to HSV images， and using threshold segmentation to RGB images based on V channel.

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
- 

### Edge Detection

- **sobel**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols)`: in `./cu/src/edgeDetection.cu`. Edge detection using sobel operator.

- **scharr**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols)`: in `./cu/src/edgeDetection.cu`. Edge detection using scharr operator.

### Erode and dilate

- **erode**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols,short int erodeElementRows,short int erodeElementCols)`: in `./cu/src/erode_dilate.cu`. 

- **dilate**`(unsigned char* dataIn,unsigned char* dataOut,short int imgRows,short int imgCols,short int dilateElementRows,short int dilateElementCols)`: in `./cu/src/erode_dilate.cu`. 

### Histogram

- **getHist**`(unsigned char* dataIn, unsigned int* hist)`： in `./cu/src/getHist.cu`. **Unfinished**
