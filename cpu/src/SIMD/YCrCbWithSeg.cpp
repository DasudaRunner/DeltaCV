//
// Created by dasuda on 19-4-3.
//

#include "deltaCV/SIMD/core.hpp"

namespace deltaCV
{
    bool ycrcbWithSeg(unsigned char *src, unsigned char *dst,const int width,const int height,
                      scalar lower,scalar upper)
    {
        assert(lower.channels()==3 && upper.channels()==3);
        int blockSize = 48; //16*3
        int block = (height * width * 3) / blockSize;

        // 加载阈值
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

        const int Shift = 15;
        const int HalfV = 1<<(Shift-1);

        //这里不能用网上给的转换公式，因为bgr转ycrcb有很多版本，我是直接看opencv源码算出的系数,
        // static const float coeffs_crb[] = { R2YF, G2YF, B2YF, YCRF, YCBF }; in /opencv-4.0.1/modules/imgproc/src/color_yuv.cpp
        //为什么要和opencv一样的系数?从结果来看，这套系数得到的cr cb通道对比度更大，也就是说红色和蓝色与其他背景区分度更高.
        const int Y_B_WT = 0.114f * (1 << Shift), Y_G_WT = 0.587f * (1 << Shift), Y_R_WT = 0.299f * (1 << Shift), Y_C_WT = 1;
        const int Cr_B_WT = 0.499f * (1 << Shift), Cr_G_WT = -0.419f * (1 << Shift), Cr_R_WT = -0.081f * (1 << Shift), Cr_C_WT = 257;
        const int Cb_B_WT = 0.395f * (1 << Shift), Cb_G_WT = -0.331f * (1 << Shift), Cb_R_WT = -0.064f * (1 << Shift), Cb_C_WT = 257;

        // load 上面的权值
        __m128i Weight_YBG = _mm_setr_epi16(Y_B_WT, Y_G_WT, Y_B_WT, Y_G_WT, Y_B_WT, Y_G_WT, Y_B_WT, Y_G_WT);
        __m128i Weight_YRC = _mm_setr_epi16(Y_R_WT, Y_C_WT, Y_R_WT, Y_C_WT, Y_R_WT, Y_C_WT, Y_R_WT, Y_C_WT);
        __m128i Weight_UBG = _mm_setr_epi16(Cr_B_WT, Cr_G_WT, Cr_B_WT, Cr_G_WT, Cr_B_WT, Cr_G_WT, Cr_B_WT, Cr_G_WT);
        __m128i Weight_URC = _mm_setr_epi16(Cr_R_WT, Cr_C_WT, Cr_R_WT, Cr_C_WT, Cr_R_WT, Cr_C_WT, Cr_R_WT, Cr_C_WT);
        __m128i Weight_VBG = _mm_setr_epi16(Cb_B_WT, Cb_G_WT, Cb_B_WT, Cb_G_WT, Cb_B_WT, Cb_G_WT, Cb_B_WT, Cb_G_WT);
        __m128i Weight_VRC = _mm_setr_epi16(Cb_R_WT, Cb_C_WT, Cb_R_WT, Cb_C_WT, Cb_R_WT, Cb_C_WT, Cb_R_WT, Cb_C_WT);
        __m128i Half = _mm_setr_epi16(0, HalfV, 0, HalfV, 0, HalfV, 0, HalfV);

        for (int i = 0; i < block; ++i, src += blockSize, dst += 16)
        {
            __m128i src1, src2, src3;

            src1 = _mm_loadu_si128((__m128i *) (src + 0)); //一次性读取个字节
            src2 = _mm_loadu_si128((__m128i *) (src + 16));
            src3 = _mm_loadu_si128((__m128i *) (src + 32));

            // 在_mm_shuffle_epi8中构造出16位的数据
            __m128i BGLL = _mm_shuffle_epi8(src1,_mm_setr_epi8(0,-1,1,-1,3,-1,4,-1,6,-1,7,-1,9,-1,10,-1));
            __m128i BGLH = _mm_shuffle_epi8(src1,_mm_setr_epi8(12,-1,13,-1,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
            BGLH = _mm_or_si128(BGLH,_mm_shuffle_epi8(src2,_mm_setr_epi8(-1,-1,-1,-1,-1,-1,0,-1,2,-1,3,-1,5,-1,6,-1)));

            __m128i BGHL = _mm_shuffle_epi8(src2, _mm_setr_epi8(8, -1, 9, -1, 11, -1, 12, -1, 14, -1, 15, -1, -1, -1, -1, -1));
            BGHL = _mm_or_si128(BGHL,_mm_shuffle_epi8(src3,_mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 2, -1)));
            __m128i BGHH = _mm_shuffle_epi8(src3, _mm_setr_epi8(4, -1, 5, -1, 7, -1, 8, -1, 10, -1, 11, -1, 13, -1, 14, -1));

            __m128i RCLL = _mm_shuffle_epi8(src1,_mm_setr_epi8(2,-1,-1,-1,5,-1,-1,-1,8,-1,-1,-1,11,-1,-1,-1));
            RCLL = _mm_or_si128(RCLL,Half);
            __m128i RCLH = _mm_shuffle_epi8(src1,_mm_setr_epi8(14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
            RCLH = _mm_or_si128(_mm_or_si128(RCLH,_mm_shuffle_epi8(src2,_mm_setr_epi8(-1,-1,-1,-1,1,-1,-1,-1,4,-1,-1,-1,7,-1,-1,-1))),Half);

            __m128i RCHL = _mm_shuffle_epi8(src2,_mm_setr_epi8(10,-1,-1,-1,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
            RCHL = _mm_or_si128(_mm_or_si128(RCHL,_mm_shuffle_epi8(src3,_mm_setr_epi8(-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,3,-1,-1,-1))),Half);
            __m128i RCHH = _mm_shuffle_epi8(src3,_mm_setr_epi8(6,-1,-1,-1,9,-1,-1,-1,12,-1,-1,-1,15,-1,-1,-1));
            RCHH = _mm_or_si128(RCHH,Half);

            // 下面是进行计算
            __m128i Result;

            __m128i Y_LL = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGLL, Weight_YBG), _mm_madd_epi16(RCLL, Weight_YRC)), Shift);
            __m128i Y_LH = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGLH, Weight_YBG), _mm_madd_epi16(RCLH, Weight_YRC)), Shift);
            __m128i Y_HL = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGHL, Weight_YBG), _mm_madd_epi16(RCHL, Weight_YRC)), Shift);
            __m128i Y_HH = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGHH, Weight_YBG), _mm_madd_epi16(RCHH, Weight_YRC)), Shift);

            __m128i Ch0 = _mm_packus_epi16(_mm_packus_epi32(Y_LL, Y_LH), _mm_packus_epi32(Y_HL, Y_HH));
            Result = _mm_cmpge_up_epu8(Ch0, ch0_min_sse);
            Result = _mm_and_si128(Result, _mm_cmpge_down_epu8(Ch0, ch0_max_sse));

            __m128i U_LL = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGLL, Weight_UBG), _mm_madd_epi16(RCLL, Weight_URC)), Shift);
            __m128i U_LH = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGLH, Weight_UBG), _mm_madd_epi16(RCLH, Weight_URC)), Shift);
            __m128i U_HL = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGHL, Weight_UBG), _mm_madd_epi16(RCHL, Weight_URC)), Shift);
            __m128i U_HH = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGHH, Weight_UBG), _mm_madd_epi16(RCHH, Weight_URC)), Shift);

            __m128i Ch1 = _mm_packus_epi16(_mm_packus_epi32(U_LL, U_LH), _mm_packus_epi32(U_HL, U_HH));
            Result = _mm_and_si128(Result, _mm_cmpge_up_epu8(Ch1, ch1_min_sse));
            Result = _mm_and_si128(Result, _mm_cmpge_down_epu8(Ch1, ch1_max_sse));

            __m128i V_LL = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGLL, Weight_VBG), _mm_madd_epi16(RCLL, Weight_VRC)), Shift);
            __m128i V_LH = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGLH, Weight_VBG), _mm_madd_epi16(RCLH, Weight_VRC)), Shift);
            __m128i V_HL = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGHL, Weight_VBG), _mm_madd_epi16(RCHL, Weight_VRC)), Shift);
            __m128i V_HH = _mm_srai_epi32(_mm_add_epi32(_mm_madd_epi16(BGHH, Weight_VBG), _mm_madd_epi16(RCHH, Weight_VRC)), Shift);

            __m128i Ch2 = _mm_packus_epi16(_mm_packus_epi32(V_LL, V_LH), _mm_packus_epi32(V_HL, V_HH));
            Result = _mm_and_si128(Result, _mm_cmpge_up_epu8(Ch2, ch2_min_sse));
            Result = _mm_and_si128(Result, _mm_cmpge_down_epu8(Ch2, ch2_max_sse));

            _mm_storeu_si128((__m128i *)dst, Result);
        }

        //剩余不足一个block的单独处理
        for (int j = blockSize * block; j < height * width; ++j, src += 3, dst++) {
            uint8_t _ch0 = src[0], _ch1 = src[1], _ch2 = src[2];

            uint8_t YUV_Y = (Y_B_WT * _ch0 + Y_G_WT * _ch1 + Y_R_WT * _ch2 + Y_C_WT * HalfV) >> Shift;
            uint8_t YUV_U = (Cr_B_WT * _ch0 + Cr_G_WT * _ch1 + Cr_R_WT * _ch2 + Cr_C_WT * HalfV) >> Shift;
            uint8_t YUV_V = (Cb_B_WT * _ch0 + Cb_G_WT * _ch1 + Cb_R_WT * _ch2 + Cb_C_WT * HalfV) >> Shift;
            if ((YUV_Y >= lower[0] && YUV_Y <= upper[0] && YUV_U >= lower[1] && YUV_U <= upper[1] && YUV_V >= lower[2] &&
                 YUV_V <= upper[2])) {
                dst[0] = 255;
            } else {
                dst[0] = 0;
            }
        }
    }
}