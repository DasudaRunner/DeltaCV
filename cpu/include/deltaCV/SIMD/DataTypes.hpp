#pragma once
#ifndef SSE_DATATYPES_HPP
#define SSE_DATATYPES_HPP

#include <assert.h>
#include <iostream>

using namespace std;

namespace deltaCV
{
    template <typename T>
    class _scalar{
    public:
        _scalar(T x);
        _scalar(T x,T y,T z);
        int channels() const;
        T operator[](const int cnt) const;
    private:
        T _val[3];
        int _ch;
    };

    template <typename T>
    _scalar<T>::_scalar(T x):_ch(1){
        _val[0] = x;
        _val[1] = 0;
        _val[2] = 0;
    }

    template <typename T>
    _scalar<T>::_scalar(T x,T y,T z):_ch(3){
        _val[0] = x;
        _val[1] = y;
        _val[2] = z;
    }

    template <typename T>
    int _scalar<T>::channels() const {
        return _ch;
    }

    template <typename T>
    T _scalar<T>::operator[](int cnt) const{
        assert(cnt < _ch && cnt>=0);
        return _val[cnt];
    }

    using scalar = _scalar<unsigned char>;
}


#endif //SSE_DATATYPES_HPP
