//
// Created by dasuda on 18-10-31.
//

#include <cudaUtils.hpp>
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
                cout<<"name: "<<prop.name<<endl;
                cout<<"maxThreadsPerBlock: "<<prop.maxThreadsPerBlock<<endl;
                cout<<"warp size: "<<prop.warpSize<<endl;
                cout<<"totalGlobalMem: "<<prop.totalGlobalMem<<endl;
                cout<<"clockRate: "<<prop.memPitch<<endl;
                cout<<"multiProcessorCount: "<<prop.multiProcessorCount<<endl;
                cout<<"maxThreadsPerMultiProcessor: "<<prop.maxThreadsPerMultiProcessor<<" 个"<<endl;
                cout<<"sharedMemPerMultiprocessor: "<<prop.sharedMemPerMultiprocessor/1024 <<" KB"<<endl;
                cout<<"major: "<<prop.major<<endl;
            }
        }
        cout<<"--------------------------------"<<endl;
        return true;
    }else {
        cout<<"[error]:No device found!"<<endl;
        return false;
    }

}