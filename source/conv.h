#ifndef _CONV_H_
#define _CONV_H_

#include "types.h"

typedef struct
{
	nnet_float_t *delta_activations;
	nnet_float_t *padded;
	size_t input_dims;
	size_t output_dims;
	size_t kernel_dims;
	size_t num_input_maps;
	size_t num_output_maps;
	activation_function_t activation_function;
} conv_layer_data_t;

layer_t *conv_create(size_t num_input_images, size_t input_dims, size_t num_output_images, size_t kernel_dims, activation_function_t func);
void conv_destroy(layer_t *layer);
void conv_forward(layer_t *layer, nnet_float_t *inputs);
void conv_backward(layer_t *layer, nnet_float_t *bperrs);
void conv_calculate_gradients(layer_t *layer, nnet_float_t *inputs);

#endif
