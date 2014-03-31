#ifndef _FFTCONV_H_
#define _FFTCONV_H_

#include "types.h"

typedef struct
{
	nnet_float_t *delta_activations;
	nnet_float_t *frequency_activations;
	nnet_float_t *frequency_kernels;
	nnet_float_t *frequency_inputs;
	nnet_float_t *frequency_errors;
	nnet_float_t *frequency_gradients;
	nnet_float_t *padded;
	size_t padded_dims;
	size_t frequency_size;
	size_t input_dims;
	size_t output_dims;
	size_t kernel_dims;
	size_t num_input_maps;
	size_t num_output_maps;
	activation_function_t activation_function;
	fftwf_plan forward, backward;
} fftconv_layer_data_t;

layer_t *fftconv_create(size_t num_input_images, size_t input_dims, size_t num_output_images, size_t kernel_dims, activation_function_t func, nnet_float_t weight_size);
void fftconv_destroy(layer_t *layer);
void fftconv_forward(layer_t *layer, nnet_float_t *inputs, int train);
void fftconv_backward(layer_t *layer, nnet_float_t *bperrs);
void fftconv_calculate_gradients(layer_t *layer, nnet_float_t *inputs);
void fftconv_start_batch(layer_t *layer);
void fftconv_end_batch(layer_t *layer);

#endif
