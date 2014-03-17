#include <float.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"
#include "maxpool.h"
#include "types.h"

layer_vtable_t maxpool_vtable;

layer_t *maxpool_create(size_t num_input_images, size_t input_dims, size_t pool_dims)
{
	layer_t *layer = (layer_t *)malloc(sizeof(layer_t));
	maxpool_layer_data_t *layer_data = (maxpool_layer_data_t *)malloc(sizeof(maxpool_layer_data_t));

	size_t output_dims = input_dims / pool_dims;
	layer->num_inputs = num_input_images * input_dims * input_dims;
	layer->num_units = num_input_images * output_dims * output_dims;
	layer->num_weights = layer->num_inputs;
	layer->weights = nnet_malloc(layer->num_weights);
	layer->activations = nnet_malloc(layer->num_units);
	layer->errors = nnet_malloc(layer->num_units);
	layer_data->input_dims = input_dims;
	layer_data->output_dims = output_dims;
	layer_data->pool_dims = pool_dims;
	layer_data->num_input_maps = num_input_images;
	layer_data->num_output_maps = num_input_images;

	memset(layer->weights, 0, sizeof(nnet_float_t) * layer->num_weights);

	maxpool_vtable.destroy = &maxpool_destroy;
	maxpool_vtable.forward = &maxpool_forward;
	maxpool_vtable.backward = &maxpool_backward;
	maxpool_vtable.calculate_gradients = 0;
	maxpool_vtable.start_batch = 0;
	maxpool_vtable.end_batch = 0;

	layer->vtable = &maxpool_vtable;
	layer->layer_data = layer_data;

	return layer;
}

void maxpool_destroy(layer_t *layer)
{
	nnet_free(layer->weights);
	nnet_free(layer->activations);
	nnet_free(layer->errors);
	free(layer->layer_data);
	free(layer);
}

void maxpool_forward(layer_t *layer, nnet_float_t *inputs)
{
	maxpool_layer_data_t *layer_data = (maxpool_layer_data_t *)layer->layer_data;
	nnet_float_t *weights = layer->weights;
	nnet_float_t *activations = layer->activations;

	//Iterate over each map
	for(size_t m = 0; m < layer_data->num_output_maps; m++)
	{
		//Iterate over each pool in the y direction
		for(size_t y = 0; y < layer_data->output_dims; y++)
		{
			for(size_t x = 0; x < layer_data->output_dims; x++)
			{
				size_t maxind = y * layer_data->pool_dims * layer_data->input_dims + x * layer_data->pool_dims;
				nnet_float_t maxval = inputs[maxind];

				//Iterate over each pixel in the receptive field
				for(size_t j = 0; j < layer_data->pool_dims; j++)
				{	
					for(size_t i = 0; i < layer_data->pool_dims; i++)
					{
						size_t ind = (y * layer_data->pool_dims + j) * layer_data->input_dims + (x * layer_data->pool_dims + i);
						weights[ind] = 0.0;

						if(inputs[ind] > maxval)
						{
							maxind = ind;
							maxval = inputs[ind];
						}
					}
				}

				activations[y * layer_data->output_dims + x] = maxval;
				weights[maxind] = 1.0;
			}
		}

		activations += layer_data->output_dims * layer_data->output_dims;
		weights += layer_data->input_dims * layer_data->input_dims;
		inputs += layer_data->input_dims * layer_data->input_dims;
	}
}

void maxpool_backward(layer_t *layer, nnet_float_t *bperrs)
{
	maxpool_layer_data_t *layer_data = (maxpool_layer_data_t *)layer->layer_data;
	nnet_float_t *errs = layer->errors;
	nnet_float_t *weights = layer->weights;
	
	for(size_t m = 0; m < layer_data->num_output_maps; m++)
	{
		for(size_t y = 0; y < layer_data->output_dims; y++)
		{
			for(size_t x = 0; x < layer_data->output_dims; x++)
			{
				size_t ind = m * layer_data->input_dims * layer_data->input_dims + y * layer_data->pool_dims * layer_data->input_dims + x * layer_data->pool_dims;

				for(size_t j = 0; j < layer_data->pool_dims; j++)
				{
					for(size_t i = 0; i < layer_data->pool_dims; i++)
					{
						bperrs[ind + j * layer_data->input_dims + i] = *errs * weights[ind + j * layer_data->input_dims + i];
					}
				}

				errs++;
			}
		}
	}
}

