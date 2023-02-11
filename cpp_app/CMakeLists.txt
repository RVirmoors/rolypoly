cmake_minimum_required(VERSION 3.16)

project(TestLibtorch)

set(SOURCE_FILES
	main.cpp
)

include_directories(	
    ${CMAKE_CURRENT_SOURCE_DIR}/../libtorch/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../libtorch/include/torch/csrc/api/include
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../libtorch/lib
)

link_libraries(
#	torch.lib	# for GPU
	torch_cpu.lib
	c10.lib
)


add_executable(TestLibtorch ${SOURCE_FILES})

target_precompile_headers(TestLibtorch 
    PRIVATE 
    <torch/torch.h>
    <torch/script.h>
    <iostream>)