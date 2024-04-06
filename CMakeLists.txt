cmake_minimum_required(VERSION 3.0)

project(mosekTutorial)# 项目名称

set(CMAKE_CXX_STANDARD 11)# 设置 C++ 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

set(HEADERDIR "/home/wcl/Solvers/mosek/10.1/tools/platform/linux64x86/h")# 设置头文件目录和库文件目录变量
set(LIBDIR "/home/wcl/Solvers/mosek/10.1/tools/platform/linux64x86/bin")

include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${HEADERDIR}
)

add_executable(${PROJECT_NAME}_lp example/7.1.1_Example_LO1.cpp)
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

add_executable(${PROJECT_NAME}_milo1 example/7.7.1_Example_MILO1.cpp)
target_link_libraries(${PROJECT_NAME}_milo1 ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_milo2 example/7.7.2_Example_MILO1.cc)
target_link_libraries(${PROJECT_NAME}_milo2 ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_mico example/7.7.3_Example_MICO1.cpp)
target_link_libraries(${PROJECT_NAME}_mico ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_djc1 example/7.8.2_Example_DJC1.cpp)
target_link_libraries(${PROJECT_NAME}_djc1 ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_reopt example/7.9_Model_Parametrization_and_Reoptimization.cpp)
target_link_libraries(${PROJECT_NAME}_reopt ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_paramMdf example/7.10_Problem_Modification_and_Reoptimization.cpp)
target_link_libraries(${PROJECT_NAME}_paramMdf ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_parallel example/7.11_Parallel_optimization.cpp)
target_link_libraries(${PROJECT_NAME}_parallel ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_infeasibility example/7.12_Retrieving_infeasibility_certificates.cpp)
target_link_libraries(${PROJECT_NAME}_infeasibility ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_response example/8.1_response.cc)
target_link_libraries(${PROJECT_NAME}_response ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_callback example/8.7_callback.cc)
target_link_libraries(${PROJECT_NAME}_callback ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_svm example/11.2_primal_svm.cc)
target_link_libraries(${PROJECT_NAME}_svm ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_low_elli example/11.6_lownerjohn_ellipsoid.cc)
target_link_libraries(${PROJECT_NAME}_low_elli ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)


add_executable(${PROJECT_NAME}_portfolio_5_card example/11.1_portfolio_5_card.cc)
target_link_libraries(${PROJECT_NAME}_portfolio_5_card ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_total_variation example/11.3_total_variation.cc)
target_link_libraries(${PROJECT_NAME}_total_variation ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

# 11.4_lpt.cc
# 11.5_logistic.cc
# 11.7_sudoku.cc
# 11.8_tsp.cc
# 11.9_nearestcorr.cc
# 11.10_qcqp_sdo_relaxation.cc
add_executable(${PROJECT_NAME}_lpt example/11.4_lpt.cc)
target_link_libraries(${PROJECT_NAME}_lpt ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_logistic example/11.5_logistic.cc)
target_link_libraries(${PROJECT_NAME}_logistic ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_sudoku example/11.7_sudoku.cc)
target_link_libraries(${PROJECT_NAME}_sudoku ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_tsp example/11.8_tsp.cc)
target_link_libraries(${PROJECT_NAME}_tsp ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_nearestcorr example/11.9_nearestcorr.cc)
target_link_libraries(${PROJECT_NAME}_nearestcorr ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)

add_executable(${PROJECT_NAME}_qcqp_sdo_relaxation example/11.10_qcqp_sdo_relaxation.cc)
target_link_libraries(${PROJECT_NAME}_qcqp_sdo_relaxation ${LIBDIR}/libmosek64.so.10.1 ${LIBDIR}/libfusion64.so.10.1)
# # 设置 DLL 搜索路径 指定 可执行文件生成目录
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBDIR})