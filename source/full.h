#ifndef _FULL_H_
#define _FULL_H_

#include "types.h"

typedef struct
{
	nnet_float_t *delta_activations;
	activation_function_t activation_function;
} full_layer_data_t;

layer_t *full_create(size_t num_inputs, size_t num_units, activation_function_t func, nnet_float_t weight_size);
void full_destroy(layer_t *layer);
void full_forward(layer_t *layer, nnet_float_t *inputs, int train);
void full_backward(layer_t *layer, nnet_float_t *bperrs);
void full_calculate_gradients(layer_t *layer, nnet_float_t *inputs);

#endif
