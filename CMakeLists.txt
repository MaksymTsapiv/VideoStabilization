cmake_minimum_required(VERSION 3.15)

set(PROJECT_NAME cv_test)
project(${PROJECT_NAME} C CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(WARNINGS_AS_ERRORS OFF)

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif ()


find_package( OpenCV REQUIRED )

# Create test examples and link with our library
function(addTestExample target_name_of_example)
	# example of usage: addTestExample(targetName src/main.cpp src/my_utils.h)
	add_executable(${ARGV})
	target_include_directories("${target_name_of_example}" PRIVATE ${OpenCV_INCLUDE_DIRS})
	target_include_directories("${target_name_of_example}" PRIVATE include)
	target_link_libraries("${target_name_of_example}" ${OpenCV_LIBS})
endfunction()


addTestExample(vidstab src/main.cpp src/adapted_optical_flow.hpp)
addTestExample(ECCTest src/transformECCexample.cpp)
addTestExample(OurECC src/ourECC.cpp src/ourECCTest.cpp)
