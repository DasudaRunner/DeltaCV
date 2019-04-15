#include "deltaCV/SIMD/core.hpp"

namespace deltaCV
{
    void weightedGrayWithSeg(unsigned char *src, unsigned char *dst,const int width,const int height,scalar_d weights,
                             scalar lower,scalar upper)
    {
        assert(lower.channels()==1 && upper.channels()==1);
        int blockSize = 96; //16*3
        int block = (height * width * 3) / blockSize;

        // 加载阈值
        __m256i ch0_min_sse = _mm256_setr_epi8(lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],
                                               lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],
                                               lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],
                                               lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0]);

        __m256i ch0_max_sse = _mm256_setr_epi8(upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],
                                               upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],
                                               upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],
                                               upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0]);

        const int Shift = 15;

        const int W_B = weights[0] * (1 << Shift), W_G = weights[1] * (1 << Shift), W_R = weights[2] * (1 << Shift);

        __m256i SSE_WBG = _mm256_setr_epi16(W_B, W_G, W_B, W_G, W_B, W_G, W_B, W_G,
                                           W_B, W_G, W_B, W_G, W_B, W_G, W_B, W_G);

        __m256i SSE_WRC = _mm256_setr_epi16(W_R, 0, W_R, 0, W_R, 0, W_R, 0,
                                            W_R, 0, W_R, 0, W_R, 0, W_R, 0);

        for (int i = 0; i < block; ++i, src += blockSize, dst += 32)
        {
            __m128i src1, src2, src3,src4,src5,src6;

            src1 = _mm_loadu_si128((__m128i *) (src + 0)); //一次性读取个字节
            src2 = _mm_loadu_si128((__m128i *) (src + 16));
            src3 = _mm_loadu_si128((__m128i *) (src + 32));
            src4 = _mm_loadu_si128((__m128i *) (src + 48));
            src5 = _mm_loadu_si128((__m128i *) (src + 64));
            src6 = _mm_loadu_si128((__m128i *) (src + 80));

            // 在_mm_shuffle_epi8中构造出16位的数据
            __m128i BG0_00 = _mm_shuffle_epi8(src1,_mm_setr_epi8(0,-1,1,-1,3,-1,4,-1,6,-1,7,-1,9,-1,10,-1));
            __m128i BG0_01 = _mm_shuffle_epi8(src1,_mm_setr_epi8(12,-1,13,-1,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
            BG0_01 = _mm_or_si128(BG0_01,_mm_shuffle_epi8(src2,_mm_setr_epi8(-1,-1,-1,-1,-1,-1,0,-1,2,-1,3,-1,5,-1,6,-1)));

            __m128i BG0_10 = _mm_shuffle_epi8(src2, _mm_setr_epi8(8, -1, 9, -1, 11, -1, 12, -1, 14, -1, 15, -1, -1, -1, -1, -1));
            BG0_10 = _mm_or_si128(BG0_10,_mm_shuffle_epi8(src3,_mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, -1)));
            __m128i BG0_11 = _mm_shuffle_epi8(src3, _mm_setr_epi8(4, -1, 5, -1, 7, -1, 8, -1, 10, -1, 11, -1, 13, -1, 14, -1));

            __m128i BG1_00 = _mm_shuffle_epi8(src4,_mm_setr_epi8(0,-1,1,-1,3,-1,4,-1,6,-1,7,-1,9,-1,10,-1));
            __m128i BG1_01 = _mm_shuffle_epi8(src4,_mm_setr_epi8(12,-1,13,-1,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
            BG1_01 = _mm_or_si128(BG1_01,_mm_shuffle_epi8(src5,_mm_setr_epi8(-1,-1,-1,-1,-1,-1,0,-1,2,-1,3,-1,5,-1,6,-1)));

            __m128i BG1_10 = _mm_shuffle_epi8(src5, _mm_setr_epi8(8, -1, 9, -1, 11, -1, 12, -1, 14, -1, 15, -1, -1, -1, -1, -1));
            BG1_10 = _mm_or_si128(BG1_10,_mm_shuffle_epi8(src6,_mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, -1)));
            __m128i BG1_11 = _mm_shuffle_epi8(src6, _mm_setr_epi8(4, -1, 5, -1, 7, -1, 8, -1, 10, -1, 11, -1, 13, -1, 14, -1));

            __m256i BG0 = _mm256_combine_si128(BG0_00,BG1_00);
            __m256i BG1 = _mm256_combine_si128(BG0_01,BG1_01);
            __m256i BG2 = _mm256_combine_si128(BG0_10,BG1_10);
            __m256i BG3 = _mm256_combine_si128(BG0_11,BG1_11);

            __m128i RC0_00 = _mm_shuffle_epi8(src1,_mm_setr_epi8(2,-1,-1,-1,5,-1,-1,-1,8,-1,-1,-1,11,-1,-1,-1));
            __m128i RC0_01 = _mm_shuffle_epi8(src1,_mm_setr_epi8(14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
            RC0_01 = _mm_or_si128(RC0_01,_mm_shuffle_epi8(src2,_mm_setr_epi8(-1,-1,-1,-1,1,-1,-1,-1,4,-1,-1,-1,7,-1,-1,-1)));

            __m128i RC0_10 = _mm_shuffle_epi8(src2,_mm_setr_epi8(10,-1,-1,-1,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
            RC0_10 = _mm_or_si128(RC0_10,_mm_shuffle_epi8(src3,_mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,3,-1,-1,-1)));
            __m128i RC0_11 = _mm_shuffle_epi8(src3,_mm_setr_epi8(6,-1,-1,-1,9,-1,-1,-1,12,-1,-1,-1,15,-1,-1,-1));

            __m128i RC1_00 = _mm_shuffle_epi8(src4,_mm_setr_epi8(2,-1,-1,-1,5,-1,-1,-1,8,-1,-1,-1,11,-1,-1,-1));

            __m128i RC1_01 = _mm_shuffle_epi8(src4,_mm_setr_epi8(14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
            RC1_01 = _mm_or_si128(RC1_01,_mm_shuffle_epi8(src5,_mm_setr_epi8(-1,-1,-1,-1,1,-1,-1,-1,4,-1,-1,-1,7,-1,-1,-1)));

            __m128i RC1_10 = _mm_shuffle_epi8(src5,_mm_setr_epi8(10,-1,-1,-1,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
            RC1_10 = _mm_or_si128(RC1_10,_mm_shuffle_epi8(src6,_mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,3,-1,-1,-1)));
            __m128i RC1_11 = _mm_shuffle_epi8(src6,_mm_setr_epi8(6,-1,-1,-1,9,-1,-1,-1,12,-1,-1,-1,15,-1,-1,-1));

            __m256i RC0 = _mm256_combine_si128(RC0_00,RC1_00);
            __m256i RC1 = _mm256_combine_si128(RC0_01,RC1_01);
            __m256i RC2 = _mm256_combine_si128(RC0_10,RC1_10);
            __m256i RC3 = _mm256_combine_si128(RC0_11,RC1_11);

            // 下面是进行计算
            __m256i Result;

            __m256i gray0 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(BG0, SSE_WBG), _mm256_madd_epi16(RC0, SSE_WRC)), Shift);
            __m256i gray1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(BG1, SSE_WBG), _mm256_madd_epi16(RC1, SSE_WRC)), Shift);
            __m256i gray2 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(BG2, SSE_WBG), _mm256_madd_epi16(RC2, SSE_WRC)), Shift);
            __m256i gray3 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(BG3, SSE_WBG), _mm256_madd_epi16(RC3, SSE_WRC)), Shift);

            __m256i packed_gray = _mm256_packus_epi16(_mm256_packus_epi32(gray0, gray1), _mm256_packus_epi32(gray2, gray3));
            Result = _mm256_cmpge_up_epu8(packed_gray, ch0_min_sse);
            Result = _mm256_and_si256(Result, _mm256_cmpge_down_epu8(packed_gray, ch0_max_sse));

            _mm256_storeu_si256((__m256i *)dst, Result);
        }
        //剩余不足一个block的单独处理
        for (int j = blockSize * block; j < height * width; ++j, src += 3, dst++) {
            uint8_t _ch0 = src[0], _ch1 = src[1], _ch2 = src[2];

            uint8_t gray_ = (W_B * _ch0 + W_G * _ch1 + W_R * _ch2) >> Shift;
            if (gray_ >= lower[0] && gray_ <= upper[0]) {
                dst[0] = 255;
            } else {
                dst[0] = 0;
            }
        }

    }

}