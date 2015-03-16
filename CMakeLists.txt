cmake_minimum_required(VERSION 2.6)

project(nnet CXX)

option(ENABLE_AVX "Enables the use of AVX instructions." OFF)
option(ENABLE_FMA "Enables the use of FMA instructions." OFF)

if(ENABLE_AVX)
	add_definitions(-DHAVE_AVX -mavx)
endif()

if(ENABLE_FMA)
	add_definitions(-DHAVE_FMA -mfma)
endif()

add_definitions(-O3 -Wall)
add_definitions(-std=c++11)

include_directories(
	include
)

add_library(
	nnet
	source/core.cpp
	source/ActivationFunction.cpp
	source/Convolutional.cpp
	source/Dropout.cpp
	source/FeedForward.cpp
	source/FullyConnected.cpp
	source/Layer.cpp
	source/Loss.cpp
	source/MaxPool.cpp
	source/SlidingFullyConnected.cpp
	source/SpatialConvolutional.cpp
	source/UpdateRule.cpp
	source/vector.cpp
)

install(TARGETS nnet DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY include/nnet DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
