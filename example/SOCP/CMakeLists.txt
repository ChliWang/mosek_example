cmake_minimum_required(VERSION 2.8.3)

project(socp)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")

find_package(Eigen3 REQUIRED)
set(HEADERDIR "/home/wcl/Solvers/mosek/10.1/tools/platform/linux64x86/h")
set(LIBDIR "/home/wcl/Solvers/mosek/10.1/tools/platform/linux64x86/bin")

include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${HEADERDIR}
)

add_executable(${PROJECT_NAME}_example example/socp_example.cpp)

add_executable(${PROJECT_NAME}_Newton_example example/socp_Newton_example.cpp)

add_executable(${PROJECT_NAME}_mosek example/socp_mosek.cpp)
target_link_libraries(${PROJECT_NAME}_example ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)


