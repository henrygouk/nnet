cmake_minimum_required(VERSION 2.8)

project(nnet C)

option(ENABLE_AVX "Enables the use of AVX instructions." OFF)

if(ENABLE_AVX)
	add_definitions(-DHAVE_AVX -mavx)
endif()

add_definitions(-g)

add_library(
	nnet
	source/conv.c
	source/core.c
	source/ffnn.c
	source/fftconv.c
	source/full.c
	source/layer.c
	source/maxpool.c
	source/vector.c
)

add_executable(
	linear
	tests/linear.c
	tests/mnist.c
)

target_link_libraries(
	linear
	nnet
	fftw3f
	m
)

add_executable(
	vector
	tests/vector.c
)

target_link_libraries(
	vector
	nnet
	fftw3f
	m
)
