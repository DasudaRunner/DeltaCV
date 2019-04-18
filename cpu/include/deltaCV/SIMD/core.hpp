#ifndef SSE_YCRCBWITHSEG_HPP
#define SSE_YCRCBWITHSEG_HPP

#include "deltaCV/SIMD/DataTypes.hpp"
#include <mmintrin.h>//MMX
#include <xmmintrin.h>//SSE
#include <emmintrin.h>//SSE2
#include <immintrin.h>//AVX
#include <smmintrin.h>
#include <pmmintrin.h>//SSE3
#include <iostream>

#define _mm_cmpge_up_epu8(a, b) _mm_cmpeq_epi8(_mm_max_epu8(a, b), a) //大于等于的留下
#define _mm_cmpge_down_epu8(a, b) _mm_cmpeq_epi8(_mm_min_epu8(a, b), a) //小于等于的留下

#define _mm_cmpgt_up_epu8(a,b) _mm_andnot_si128(_mm_cmpeq_epi8(a,b),_mm_cmpeq_epi8(_mm_max_epu8(a,b),a)) //大于的留下

#define _mm256_cmpge_up_epu8(a, b) _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a) //大于等于的留下 avx2
#define _mm256_cmpge_down_epu8(a, b) _mm256_cmpeq_epi8(_mm256_min_epu8(a, b), a) //小于等于的留下

#define _mm256_combine_si128(a,b) _mm256_insertf128_si256(_mm256_castsi128_si256(a),b,1) //将两个__mm128i 拼接成一个__mm256i a为low b为high

namespace deltaCV
{
    using namespace std;
    /**  @brief: more details - >　https://dasuda.top/deltacv/2019/04/02/DeltaCV%E4%B9%8BCPU%E7%AE%97%E6%B3%95%E4%BC%98%E5%8C%96-inRange/
     *
     * src: input 3 channels or 1 channel
     * dst: output
     * width:
     * height:
     * lower: lower boundary
     * upper: upper boundary
     */
    void inRange(unsigned char *src, unsigned char *dst, int width, int height,
                 scalar lower, scalar upper);

    /**  @brief: more details - > https://dasuda.top/deltacv/2019/04/02/DeltaCV%E4%B9%8BCPU%E7%AE%97%E6%B3%95%E4%BC%98%E5%8C%96-YCrCb%E7%A9%BA%E9%97%B4%E9%98%88%E5%80%BC%E5%88%86%E5%89%B2/　
     *
     * src: input(BGR image)
     * dst: output
     * width:
     * height:
     * lower: lower boundary: 3 channels
     * upper: upper boundary: 3 channels
     */
    bool ycrcbWithSeg(unsigned char *src, unsigned char *dst,const int width,const int height,
                      scalar lower,scalar upper);

    /**  @brief: more details - >
     *
     * src: input(BGR image)
     * dst: output
     * width:
     * height:
     * lower: lower boundary: 1 channels
     * upper: upper boundary: 1 channels
     */
    void weightedGrayWithSeg(unsigned char *src, unsigned char *dst,const int width,const int height,scalar_d weights,
                             scalar lower,scalar upper);


    /**  @brief: more details - >
     *
     * src: input(BGR image)
     * dst1: output of grayscale
     * dst2: output of bitwise_and
     * width:
     * height:
     * gray_weights: deltaCV::scalar_d(0.144,0.587,0.299) for normal gray scale
     * color_mode： 0-> red, 1-> blue
     * thres1: grayscale threshold
     * thres2: b/r-g threshold
     */
    void grayBRWithSeg(unsigned char *src, unsigned char *dst1,unsigned char *dst2,const int width,const int height,
                       scalar_d gray_weights,
                       const unsigned char color_mode,
                       scalar thres1,scalar thres2);

}


#endif //SSE_YCRCBWITHSEG_HPP
