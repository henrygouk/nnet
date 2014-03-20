#include <stdlib.h>
#include <string.h>

#include "core.h"
#include "full.h"
#include "vector.h"

layer_vtable_t full_vtable;

layer_t *full_create(size_t num_inputs, size_t num_units, activation_function_t func)
{
	layer_t *layer = (layer_t *)malloc(sizeof(layer_t));
	full_layer_data_t *layer_data = (full_layer_data_t *)malloc(sizeof(full_layer_data_t));

	layer->num_inputs = num_inputs;
	layer->num_units = num_units;
	layer->num_weights = num_inputs * num_units + num_units;
	layer->weights = nnet_malloc(layer->num_weights);
	layer->gradients = nnet_malloc(layer->num_weights);
	layer->activations = nnet_malloc(layer->num_units);
	layer->errors = nnet_malloc(layer->num_units);
	layer_data->delta_activations = nnet_malloc(layer->num_units);
	layer_data->activation_function = func;

	random_vector(layer->weights, layer->num_weights, -0.1, 0.1);
	memset(layer->gradients, 0, sizeof(nnet_float_t) * layer->num_weights);

	full_vtable.destroy = &full_destroy;
	full_vtable.forward = &full_forward;
	full_vtable.backward = &full_backward;
	full_vtable.calculate_gradients = &full_calculate_gradients;
	full_vtable.start_batch = 0;
	full_vtable.end_batch = 0;

	layer->vtable = &full_vtable;
	layer->layer_data = layer_data;

	return layer;
}

void full_destroy(layer_t *layer)
{
	full_layer_data_t *layer_data = (full_layer_data_t *)layer->layer_data;

	nnet_free(layer->weights);
	nnet_free(layer->gradients);
	nnet_free(layer->activations);
	nnet_free(layer->errors);
	nnet_free(layer_data->delta_activations);
	free(layer_data);
	free(layer);
}

void full_forward(layer_t *layer, nnet_float_t *inputs)
{
	full_layer_data_t *layer_data = (full_layer_data_t *)layer->layer_data;

	matrix_vector_mul(layer->weights, layer->num_units, layer->num_inputs, inputs, layer->activations);
	vector_accum(layer->activations, layer->weights + layer->num_units * layer->num_inputs, layer->num_units);

	layer_calculate_activations(layer, layer_data->delta_activations, layer_data->activation_function);
}

void full_backward(layer_t *layer, nnet_float_t *bperrs)
{
	matrix_trans_vector_mul(layer->weights, layer->num_units, layer->num_inputs, layer->errors, bperrs);
}

void full_calculate_gradients(layer_t *layer, nnet_float_t *inputs)
{
	full_layer_data_t *layer_data = (full_layer_data_t *)layer->layer_data;

	//Finish calculating the errors wrt to each unit
	vector_mul(layer->errors, layer_data->delta_activations, layer->errors, layer->num_units);

	//Now calculate the error wrt to each weight
	for(size_t u = 0; u < layer->num_units; u++)
	{
		vector_scale_accum(layer->gradients + u * layer->num_inputs, inputs, layer->errors[u], layer->num_inputs);
	}

	vector_accum(layer->gradients + layer->num_units * layer->num_inputs, layer->errors, layer->num_units);
}
