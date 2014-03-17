cmake_minimum_required(VERSION 2.8)

project(nnet C)

option(ENABLE_AVX "Enables the use of AVX instructions." OFF)

if(ENABLE_AVX)
	add_definitions(-DHAVE_AVX -mavx)
endif()

add_definitions(-g -O3)

add_library(
	nnet
	source/conv.c
	source/core.c
	source/ffnn.c
	source/fftconv.c
	source/full.c
	source/layer.c
	source/maxpool.c
	source/update.c
	source/vector.c
)

add_executable(
	mnist_cnn
	tests/mnist_cnn.c
	tests/mnist.c
)

target_link_libraries(
	mnist_cnn
	nnet
	fftw3f
	m
)

add_executable(
	mnist_mlp
	tests/mnist_mlp.c
	tests/mnist.c
)

target_link_libraries(
	mnist_mlp
	nnet
	fftw3f
	m
)

add_executable(
	mnist_linear
	tests/mnist_linear.c
	tests/mnist.c
)

target_link_libraries(
	mnist_linear
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
