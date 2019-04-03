#include "deltaCV/SIMD/core.hpp"

namespace deltaCV
{
    int inRange(unsigned char *src, unsigned char *dst, int width, int height,
                     scalar lower, scalar upper)
    {
        assert(lower.channels()==upper.channels());
        int channel = lower.channels();
        if(channel==3)
        {
            int blockSize = 16; //16*3
            int block = height * width  / blockSize;

            __m128i ch0_min_sse = _mm_setr_epi8(lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],
                                                lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0]);
            __m128i ch1_min_sse = _mm_setr_epi8(lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],
                                                lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1],lower[1]);
            __m128i ch2_min_sse = _mm_setr_epi8(lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],
                                                lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2],lower[2]);

            __m128i ch0_max_sse = _mm_setr_epi8(upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],
                                                upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0]);
            __m128i ch1_max_sse = _mm_setr_epi8(upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],
                                                upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1],upper[1]);
            __m128i ch2_max_sse = _mm_setr_epi8(upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],
                                                upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2],upper[2]);

            for(int i=0; i<block; ++i,src+=blockSize*channel,dst+=blockSize)
            {
                // 读取数据
                __m128i src1 = _mm_loadu_si128((__m128i *) (src + 0));
                __m128i src2 = _mm_loadu_si128((__m128i *) (src + blockSize*1));
                __m128i src3 = _mm_loadu_si128((__m128i *) (src + blockSize*2));

                //三通道分离
                __m128i Ch0 = _mm_shuffle_epi8(src1, _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
                Ch0 = _mm_or_si128(Ch0, _mm_shuffle_epi8(src2,
                                                         _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1,
                                                                       -1, -1, -1)));
                Ch0 = _mm_or_si128(Ch0, _mm_shuffle_epi8(src3,
                                                         _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4,
                                                                       7, 10, 13)));
                __m128i Ch1 = _mm_shuffle_epi8(src1,
                                       _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
                Ch1 = _mm_or_si128(Ch1, _mm_shuffle_epi8(src2,
                                                         _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1,
                                                                       -1, -1, -1)));
                Ch1 = _mm_or_si128(Ch1, _mm_shuffle_epi8(src3,
                                                         _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5,
                                                                       8, 11, 14)));

                __m128i Ch2 = _mm_shuffle_epi8(src1,
                                       _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
                Ch2 = _mm_or_si128(Ch2, _mm_shuffle_epi8(src2,
                                                         _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1,
                                                                       -1, -1, -1)));
                Ch2 = _mm_or_si128(Ch2, _mm_shuffle_epi8(src3,
                                                         _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6,
                                                                       9, 12, 15)));
                // 阈值判断
                __m128i Result;
                //ch0
                Result = _mm_cmpge_up_epu8(Ch0, ch0_min_sse);
                Result = _mm_and_si128(Result, _mm_cmpge_down_epu8(Ch0, ch0_max_sse));
                //ch1
                Result = _mm_and_si128(Result, _mm_cmpge_up_epu8(Ch1, ch1_min_sse));
                Result = _mm_and_si128(Result, _mm_cmpge_down_epu8(Ch1, ch1_max_sse));
                //ch2
                Result = _mm_and_si128(Result, _mm_cmpge_up_epu8(Ch2, ch2_min_sse));
                Result = _mm_and_si128(Result, _mm_cmpge_down_epu8(Ch2, ch2_max_sse));

                //输出
                _mm_storeu_si128((__m128i *) (dst + 0), Result);
            }
            //剩余不足一个block的单独处理
            for (int j = blockSize * block * channel; j < height * width; ++j, src += channel, dst++) {
                uint8_t _ch0 = src[0], _ch1 = src[1], _ch2 = src[2];
                if (_ch0 >= lower[0] && _ch0 <= upper[0] && _ch1 >= lower[1] && _ch1 <= upper[1] && _ch2 >= lower[2] &&
                     _ch2 <= upper[2]) {
                    dst[0] = 255;
                } else {
                    dst[0] = 0;
                }
            }

        }else
        {
            int blockSize = 32;
            int block = height * width  / blockSize;

            __m256i ch0_min_sse = _mm256_setr_epi8(lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],
                                                   lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],
                                                   lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],
                                                   lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0],lower[0]);

            __m256i ch0_max_sse = _mm256_setr_epi8(upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],
                                                   upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],
                                                   upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],
                                                   upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0],upper[0]);

            for(int i=0; i<block; ++i,src+=blockSize,dst+=blockSize)
            {
                __m256i src1 = _mm256_loadu_si256((__m256i *) (src + 0));
                __m256i Result;
                Result = _mm256_cmpge_up_epu8(src1, ch0_min_sse);
                Result = _mm256_and_si256(Result, _mm256_cmpge_down_epu8(src1, ch0_max_sse));
                _mm256_storeu_si256((__m256i *) (dst + 0), Result);
            }

            for (int j = blockSize * block * channel; j < height * width; ++j, src += channel, dst++) {
                uint8_t _ch0 = src[0], _ch1 = src[1], _ch2 = src[2];
                if (_ch0 >= lower[0] && _ch0 <= upper[0] && _ch1 >= lower[1] && _ch1 <= upper[1] && _ch2 >= lower[2] &&
                    _ch2 <= upper[2]) {
                    dst[0] = 255;
                } else {
                    dst[0] = 0;
                }
            }
        }

    }
}
