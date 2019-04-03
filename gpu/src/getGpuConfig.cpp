#include "deltaCV/gpu/cudaUtils.hpp"
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

bool getGPUConfig()
{
    int count;
    cudaGetDeviceCount(&count);

    cout<<"Supported GPU: "<<count<<endl;

    if(count>0)
    {
        for (int j = 0; j < count; ++j) {
            cout<<"-------------GPU["<<j<<"]-------------"<<endl;
            cudaDeviceProp prop;
            if(cudaGetDeviceProperties(&prop,j)==cudaSuccess){
                cout<<"Name: "<<prop.name<<endl;
                cout<<"MaxThreadsPerBlock: "<<prop.maxThreadsPerBlock<<endl;
                cout<<"Warp size: "<<prop.warpSize<<endl;
                cout<<"TotalGlobalMem: "<<prop.totalGlobalMem<<endl;
                cout<<"ClockRate: "<<prop.memPitch<<endl;
                cout<<"MultiProcessorCount: "<<prop.multiProcessorCount<<endl;
                cout<<"MaxThreadsPerMultiProcessor: "<<prop.maxThreadsPerMultiProcessor<<" 个"<<endl;
                cout<<"SharedMemPerMultiprocessor: "<<prop.sharedMemPerMultiprocessor/1024 <<" KB"<<endl;
                cout<<"Compute capability: "<<prop.major<<"."<<prop.minor<<endl;
            }
        }
        cudaSetDevice(count);
        cout<<"--------------------------------"<<endl;
        return true;
    }else {
        cout<<"[error]:No device found!"<<endl;
        return false;
    }

}
