#include "deltaCV/SIMD/DataTypes.hpp"
namespace deltaCV
{
    scalar::scalar(int x):_ch(1){
        _val[0] = x;
        _val[1] = 0;
        _val[2] = 0;
    }
    scalar::scalar(int x,int y,int z):_ch(3){
        _val[0] = x;
        _val[1] = y;
        _val[2] = z;
    }
    int scalar::channels() const {
        return _ch;
    }
    int scalar::operator[](int cnt) const{
        assert(cnt < _ch && cnt>=0);
        return _val[cnt];
    }
}

