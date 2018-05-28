cmake_minimum_required(VERSION 2.6)
project(Graph_coloring_seq)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -pedantic -Wextra -Wno-unused-variable")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ")

add_executable(Main main.cpp template_cpu.h)
add_executable(Main2 main_moze_lepsze.cpp template_cpu.h)
