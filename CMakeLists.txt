cmake_minimum_required(VERSION 2.6)

project(nnet CXX)

option(ENABLE_AVX "Enables the use of AVX instructions." OFF)

if(ENABLE_AVX)
	add_definitions(-DHAVE_AVX -mavx -mavx2 -mfma)
endif()

add_definitions(-g -O3 -Wall)
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

add_executable(
	mincnn
	util/mincnn.cpp
	util/evaluate.cpp
)

target_link_libraries(
	mincnn
	nnet
	fftw3f
)

add_executable(
	showfilters
	util/showfilters.cpp
)

target_link_libraries(
	showfilters
	pthread
	X11
)

add_executable(
	mnistcnn
	util/mnistcnn.cpp
	util/mnist.cpp
	util/evaluate.cpp
)

target_link_libraries(
	mnistcnn
	nnet
	fftw3f
)

add_executable(
	corelcbir
	util/corelcbir.cpp
	util/evaluate.cpp
)

target_link_libraries(
	corelcbir
	nnet
	fftw3f
	pthread
	X11
)

add_executable(
	corelvecs
	util/corelvecs.cpp
)

target_link_libraries(
	corelvecs
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

install(TARGETS nnet DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY include/nnet DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
