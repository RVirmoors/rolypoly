cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(backend)

list(APPEND CMAKE_PREFIX_PATH "E:/GitHub/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(${PROJECT_NAME} INTERFACE)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_11)

target_include_directories(
    ${PROJECT_NAME}
    INTERFACE
    $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(backend INTERFACE "${TORCH_LIBRARIES}")
set_property(TARGET backend PROPERTY CXX_STANDARD 14)

add_executable(backend_test test/test.cpp)
target_include_directories(backend_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(backend_test "${TORCH_LIBRARIES}")
set_property(TARGET backend_test PROPERTY CXX_STANDARD 14)

target_precompile_headers(${PROJECT_NAME} 
	INTERFACE 
	<torch/torch.h>
	<iostream>
	<vector>)

target_precompile_headers(backend_test 
	PRIVATE 
	<torch/torch.h>
	<iostream>
	<vector>)
