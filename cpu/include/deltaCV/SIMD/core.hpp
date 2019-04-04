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

    /**  @brief: more details - >　
     *
     * src: input(BGR image)
     * dst: output
     * width:
     * height:
     * lower: lower boundary
     * upper: upper boundary
     */
    bool ycrcbWithSeg(unsigned char *src, unsigned char *dst,const int width,const int height,
                      scalar lower,scalar upper);
}


#endif //SSE_YCRCBWITHSEG_HPP
