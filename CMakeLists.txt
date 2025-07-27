cmake_minimum_required(VERSION 3.20)

project(gpu_fft VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(gpu_fft gpu_fft.cc)

if(MSVC)
    target_compile_options(gpu_fft PRIVATE /W4)
else()
    target_compile_options(gpu_fft PRIVATE -Wall -Wextra -Wpedantic -O2)
endif()

