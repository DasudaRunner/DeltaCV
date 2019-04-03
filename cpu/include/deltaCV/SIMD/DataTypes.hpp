#pragma once
#ifndef SSE_DATATYPES_HPP
#define SSE_DATATYPES_HPP
#include <assert.h>
#include <iostream>
using namespace std;
namespace deltaCV
{
    class scalar{
    public:
        scalar(int x);
        scalar(int x,int y,int z);
        int channels() const;
        int operator[](const int cnt) const;
    private:
        int _val[3];
        int _ch;
    };
}


#endif //SSE_DATATYPES_HPP
