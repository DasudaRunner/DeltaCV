cmake_minimum_required(VERSION 3.5)
project(nodeProcess)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/../)

find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS system atomic)#interprocess不是以库的形式链接的，直接是头文件的形式，所以此处并不需要
find_package(OpenCV REQUIRED)

cuda_add_executable(nodeProcess
        nodeProcess.cpp
        )

#link_directories(testcmake ${Boost_LIBRARY_DIRS})
#因为managed_shared_memory需要加上-pthread
target_link_libraries(nodeProcess ${Boost_LIBRARIES} ${OpenCV_LIBS} rt -pthread) #这个后面需要加上rt
