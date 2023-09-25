# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.0)
project(rolypoly_tilde)

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

#############################################################
# MAX EXTERNAL
#############################################################


include_directories( 
	"${C74_INCLUDES}"
	${CMAKE_CURRENT_SOURCE_DIR}/../../../midifile/include
	${CMAKE_CURRENT_SOURCE_DIR}/../../../../libtorch/include
	${CMAKE_CURRENT_SOURCE_DIR}/../../../../libtorch/include/torch/csrc/api/include
	${CMAKE_CURRENT_SOURCE_DIR}/../backend/include
)


if (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../libtorch_x86")
else()
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../libtorch")
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set( SOURCE_FILES
	${PROJECT_NAME}.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/../../../midifile/src/MidiFile.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/../../../midifile/src/MidiEvent.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/../../../midifile/src/MidiMessage.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/../../../midifile/src/MidiEventList.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/../../../midifile/src/Binasc.cpp
)

add_library( 
	${PROJECT_NAME} 
	MODULE
	${SOURCE_FILES}
)

target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_CUDA_LIBRARIES}")

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
if (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
	add_custom_command(
		TARGET ${PROJECT_NAME}  POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E rename
		"${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo"
		"${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}_x86.mxo"
	COMMENT "Renamed to _x86.mxo")
endif()
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
endif()
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")

#############################################################
# UNIT TEST
#############################################################

# include(${C74_MIN_API_DIR}/test/min-object-unittest.cmake)

target_precompile_headers(${PROJECT_NAME} 
	PRIVATE 
	<torch/torch.h>
	<string>
	<thread>
	<vector>
	<MidiFile.h>)