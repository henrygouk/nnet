cmake_minimum_required(VERSION 2.6)

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
	source/dropout.c
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
	util/mnist_cnn.c
	util/mnist.c
	util/evaluate.c
)

target_link_libraries(
	mnist_cnn
	nnet
	fftw3f
	m
)

add_executable(
	mnist_mlp
	util/mnist_mlp.c
	util/mnist.c
	util/evaluate.c
)

target_link_libraries(
	mnist_mlp
	nnet
	fftw3f
	m
)

add_executable(
	mnist_linear
	util/mnist_linear.c
	util/mnist.c
	util/evaluate.c
)

target_link_libraries(
	mnist_linear
	nnet
	fftw3f
	m
)

add_executable(
	cifar10_cnn
	util/cifar10_cnn.c
	util/cifar10.c
	util/mnist.c
	util/evaluate.c
)

target_link_libraries(
	cifar10_cnn
	nnet
	fftw3f
	m
)

add_executable(
	vector
	util/vector.c
)

target_link_libraries(
	vector
	nnet
	fftw3f
	m
)
