cmake_minimum_required(VERSION 3.10)
project(deltaCV)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
)

include_directories(
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS system atomic)#interprocess不是以库的形式链接的，直接是头文件的形式，所以此处并不需要
find_package(OpenCV REQUIRED)

file(GLOB DIR_CU_SRCS ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB DIR_CPP_SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)

cuda_add_executable(deltaCV
        main.cpp
        ${DIR_CU_SRCS}
        ${DIR_CPP_SRCS}
        )

target_link_libraries(deltaCV ${Boost_LIBRARIES} ${OpenCV_LIBS} rt -pthread)
