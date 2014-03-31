#include <float.h>
#include <stdlib.h>

#include "dropout.h"
#include "core.h"

static layer_vtable_t dropout_vtable;

layer_t *dropout_create(size_t num_units, nnet_float_t prob)
{
	layer_t *layer = malloc(sizeof(layer_t));
	dropout_layer_data_t *layer_data = malloc(sizeof(dropout_layer_data_t));

	layer_data->prob = prob;
	layer->layer_data = layer_data;
	layer->activations = nnet_malloc(num_units);
	layer->errors = nnet_malloc(num_units);
	layer->num_weights = 0;
	layer->num_inputs = num_units;
	layer->num_units = num_units;

	dropout_vtable.destroy = &dropout_destroy;
	dropout_vtable.forward = &dropout_forward;
	dropout_vtable.backward = &dropout_backward;
	dropout_vtable.calculate_gradients = 0;
	dropout_vtable.start_batch = 0;
	dropout_vtable.end_batch = 0;

	layer->vtable = &dropout_vtable;

	return layer;
}

void dropout_destroy(layer_t *layer)
{
	nnet_free(layer->activations);
	nnet_free(layer->errors);
	free(layer->layer_data);
	free(layer);
}

void dropout_forward(layer_t *layer, nnet_float_t *inputs, int train)
{
	nnet_float_t prob = ((dropout_layer_data_t *)(layer->layer_data))->prob;

	if(train)
	{
		for(size_t i = 0; i < layer->num_units; i++)
		{
			if((nnet_float_t)rand() / FLT_MAX < prob)
				layer->activations[i] = inputs[i];
			else
				layer->activations[i] = 0.0;
		}
	}
	else
	{
		for(size_t i = 0; i < layer->num_units; i++)
		{
			layer->activations[i] = inputs[i] * prob;
		}
	}
}

void dropout_backward(layer_t *layer, nnet_float_t *bperrs)
{
	for(size_t i = 0; i < layer->num_units; i++)
	{
		bperrs[i] = layer->errors[i];
	}
}
