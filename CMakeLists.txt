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

# 添加可执行文件
add_executable(${PROJECT_NAME}_lp example/LP.cpp)
add_executable(${PROJECT_NAME}_cqp example/CQP.cpp)

# 添加链接库
target_link_libraries(${PROJECT_NAME}_lp ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)
target_link_libraries(${PROJECT_NAME}_cqp ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

# 设置 DLL 搜索路径 放最后才能生成 可执行文件
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBDIR})