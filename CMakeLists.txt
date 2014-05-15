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
	source/x86/ActivationFunction.cpp
	source/x86/Convolutional.cpp
	source/x86/core.cpp
	source/x86/FeedForward.cpp
	source/x86/FullyConnected.cpp
	source/x86/Layer.cpp
	source/x86/loss.cpp
	source/x86/MaxPool.cpp
	source/x86/UpdateRule.cpp
	source/x86/vector.cpp
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
