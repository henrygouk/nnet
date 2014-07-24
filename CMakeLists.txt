cmake_minimum_required(VERSION 2.6)

project(nnet CXX)

option(ENABLE_AVX "Enables the use of AVX instructions." OFF)

if(ENABLE_AVX)
	add_definitions(-DHAVE_AVX -mavx -mavx2 -mfma)
endif()

add_definitions(-g -O3)
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
	source/UpdateRule.cpp
	source/vector.cpp
)

add_executable(
	cifar10cnn
	util/cifar10.cpp
	util/cifar10cnn.cpp
	util/evaluate.cpp
)

target_link_libraries(
	cifar10cnn
	nnet
	fftw3f
)

add_executable(
	vector
	util/vector.cpp
)

target_link_libraries(
	vector
	fftw3f
	nnet
)
