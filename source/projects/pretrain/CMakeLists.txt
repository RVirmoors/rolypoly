cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pretrain)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(pretrain pretrain.cpp)
target_include_directories(${PROJECT_NAME} 
	PRIVATE 
	${PROJECT_SOURCE_DIR}/../backend/include
)
target_link_libraries(pretrain "${TORCH_LIBRARIES}")
set_property(TARGET pretrain PROPERTY CXX_STANDARD 17)

target_precompile_headers(${PROJECT_NAME} 
	PRIVATE 
	<torch/torch.h>
	<iostream>
	<vector>)       

install(FILES $<TARGET_RUNTIME_DLLS:pretrain> TYPE BIN) # copies dll files
# see https://stackoverflow.com/questions/50137141/cmake-copy-dlls-to-the-runtime-output-directory
# https://stackoverflow.com/questions/10671916/how-to-copy-dll-files-into-the-same-folder-as-the-executable-using-cmake/34445220#34445220