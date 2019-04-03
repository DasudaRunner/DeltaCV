cmake_minimum_required(VERSION 3.5)
project(deltaCV)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=haswell")

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
add_compile_options(-O3)
find_package(OpenCV REQUIRED)


include_directories(
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/include)
find_package(Boost REQUIRED COMPONENTS system atomic)#interprocess不是以库的形式链接的，直接是头文件的形式，所以此处并不需要
find_package(OpenCV REQUIRED)

file(GLOB DIR_CPP_SRCS ${PROJECT_SOURCE_DIR}/src/SIMD/*.cpp)

add_executable(deltaCV
        main.cpp
        ${DIR_CPP_SRCS}
        )
target_link_libraries(deltaCV ${Boost_LIBRARIES} ${OpenCV_LIBS} rt -pthread)
