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
    /**  @brief: more details - >
     *
     * src: input
     * dst: output
     * width:
     * height:
     * lower: lower boundary
     * upper: upper boundary
     */
    int inRange(unsigned char *src, unsigned char *dst, int width, int height,
                     scalar lower, scalar upper);
}


#endif //SSE_YCRCBWITHSEG_HPP
