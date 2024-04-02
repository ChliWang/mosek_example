cmake_minimum_required(VERSION 3.0)

# 项目名称
project(mosekTutorial)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# # 设置运行时库为 /MD
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MD")

# 设置头文件目录和库文件目录变量
set(HEADERDIR "/home/wcl/Solvers/mosek/10.1/tools/platform/linux64x86/h")
set(LIBDIR "/home/wcl/Solvers/mosek/10.1/tools/platform/linux64x86/bin")

# 添加头文件目录
include_directories(${HEADERDIR})

# Linear_Optimization
# Conic_Quadratic_Optimization
# Power_Cone_Optimization
# Geometric_Programming.cpp

# 添加可执行文件
add_executable(${PROJECT_NAME}_lp example/7.1.1_Example_LO1.cpp)
# 添加链接库
target_link_libraries(${PROJECT_NAME}_lp ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_cqp example/7.2.1_Example_CQO1.cpp)
target_link_libraries(${PROJECT_NAME}_cqp ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_pco example/7.3.1_Example_POW1.cpp)
target_link_libraries(${PROJECT_NAME}_pco ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_gp example/7.5.1_Example_GP1.cpp)
target_link_libraries(${PROJECT_NAME}_gp ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_sdp1 example/7.6.1_Example_SDO1.cpp)
target_link_libraries(${PROJECT_NAME}_sdp1 ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_sdp2 example/7.6.1_Example_SDO2.cpp)
target_link_libraries(${PROJECT_NAME}_sdp2 ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_sdp3 example/7.6.1_Example_SDO3.cpp)
target_link_libraries(${PROJECT_NAME}_sdp3 ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_milo example/7.7.1_Example_MILO1.cpp)
target_link_libraries(${PROJECT_NAME}_milo ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

# 7.7.3 是设定初始迭代值

add_executable(${PROJECT_NAME}_mico example/7.7.3_Example_MICO1.cpp)
target_link_libraries(${PROJECT_NAME}_mico ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

# # 设置 DLL 搜索路径 指定 可执行文件生成目录
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBDIR})