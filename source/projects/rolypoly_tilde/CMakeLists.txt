# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.0)

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

#############################################################
# MAX EXTERNAL
#############################################################


include_directories( 
	"${C74_INCLUDES}"
	${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include
	${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/include/torch/csrc/api/include
)

link_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch/lib
	${CMAKE_CURRENT_SOURCE_DIR}/../../../nn_tilde/build/backend/Release
)

link_libraries(
#	torch.lib	# for GPU
	torch_cpu.lib
	c10.lib
	backend.lib
)


set( SOURCE_FILES
	${PROJECT_NAME}.cpp
)


add_library( 
	${PROJECT_NAME} 
	MODULE
	${SOURCE_FILES}
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

if (APPLE) # SEARCH FOR TORCH DYLIB IN THE LOADER FOLDER
set_target_properties(${PROJECT_NAME} PROPERTIES
BUILD_WITH_INSTALL_RPATH FALSE
LINK_FLAGS "-Wl,-rpath,@loader_path/"
)
endif()

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

if (APPLE) # COPY TORCH DYLIB IN THE LOADER FOLDER
add_custom_command( 
		TARGET ${PROJECT_NAME} 
		POST_BUILD 
		COMMAND cp "${TORCH_INSTALL_PREFIX}/lib/*.dylib" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo/Contents/MacOS/" 
		COMMENT "Copy Torch Libraries" 
	)

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
	add_custom_command( 
		TARGET ${PROJECT_NAME} 
		POST_BUILD 
		COMMAND "codesign" "--force" "--deep" "-s" "-" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo"
		COMMENT "Codesign external" 
	)
endif()
endif()

#############################################################
# UNIT TEST
#############################################################

# include(${C74_MIN_API_DIR}/test/min-object-unittest.cmake)
