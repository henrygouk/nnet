#include <stdlib.h>
#include <string.h>

#include "conv.h"
#include "core.h"
#include "vector.h"

layer_vtable_t conv_vtable;

layer_t *conv_create(size_t num_input_images, size_t input_dims, size_t num_output_images, size_t kernel_dims, activation_function_t func)
{
	layer_t *layer = (layer_t *)malloc(sizeof(layer_t));

	size_t output_dims = input_dims - kernel_dims + 1;
	layer->num_inputs = num_input_images * input_dims * input_dims;
	layer->num_units = num_output_images * output_dims * output_dims;
	layer->num_weights = num_input_images * num_output_images * kernel_dims * kernel_dims + num_output_images;
	layer->weights = nnet_malloc(layer->num_weights);
	layer->gradients = nnet_malloc(layer->num_weights);
	layer->activations = nnet_malloc(layer->num_units);
	layer->errors = nnet_malloc(layer->num_units);
	layer->delta_activations = nnet_malloc(layer->num_units);
	layer->input_dims = input_dims;
	layer->output_dims = output_dims;
	layer->kernel_dims = kernel_dims;
	layer->num_input_maps = num_input_images;
	layer->num_output_maps = num_output_images;
	layer->activation_function = func;

	random_vector(layer->weights, layer->num_weights, -0.01, 0.01);
	memset(layer->gradients, 0, sizeof(nnet_float_t) * layer->num_weights);

	conv_vtable.destroy = &conv_destroy;
	conv_vtable.forward = &conv_forward;
	conv_vtable.backward = &conv_backward;
	conv_vtable.calculate_gradients = &conv_calculate_gradients;
	conv_vtable.update = &conv_update;

	layer->vtable = &conv_vtable;

	return layer;
}

void conv_destroy(layer_t *layer)
{
	nnet_free(layer->weights);
	nnet_free(layer->gradients);
	nnet_free(layer->activations);
	nnet_free(layer->errors);
	nnet_free(layer->delta_activations);
	free(layer);
}

void conv_forward(layer_t *layer, nnet_float_t *inputs)
{
	nnet_float_t *output = layer->activations;
	nnet_float_t *weights = layer->weights;

	//Iterate over each output map
	for(size_t u = 0; u < layer->num_output_maps; u++)
	{
		//Set all pixels for this output map to the bias value
		for(size_t p = 0; p < layer->output_dims * layer->output_dims; p++)
		{
			output[p] = layer->weights[layer->num_weights - layer->num_output_maps + u];
		}

		nnet_float_t *input = inputs;

		//iterate over each input map
		for(size_t i = 0; i < layer->num_input_maps; i++)
		{
			convolve_valid(input, layer->input_dims, weights, layer->kernel_dims, output);

			inputs += layer->input_dims * layer->input_dims;
			weights += layer->kernel_dims * layer->kernel_dims;
		}
		
		output += layer->output_dims * layer->output_dims;
	}

	layer_calculate_activations(layer);
}

void conv_backward(layer_t *layer, nnet_float_t *bperrs)
{
	const size_t kernel_size = layer->kernel_dims * layer->kernel_dims;
	const size_t input_size = layer->input_dims * layer->input_dims;
	const size_t output_size = layer->output_dims * layer->output_dims;

	for(size_t i = 0; i < layer->num_input_maps; i++)
	{
		nnet_float_t *kernels = layer->weights + i * kernel_size;

		for(size_t o = 0; o < layer->num_output_maps; o++)
		{
			correlate_full(layer->errors + o * output_size, layer->output_dims, kernels, layer->kernel_dims, bperrs);

			kernels += kernel_size * layer->num_input_maps;
		}

		bperrs += input_size;
	}
}

void conv_calculate_gradients(layer_t *layer, nnet_float_t *inputs)
{
	const size_t kernel_size = layer->kernel_dims * layer->kernel_dims;
	const size_t input_size = layer->input_dims * layer->input_dims;
	const size_t output_size = layer->output_dims * layer->output_dims;
	nnet_float_t *errors = layer->errors;
	nnet_float_t *deltas = layer->delta_activations;
	nnet_float_t *gradients = layer->gradients;

	for(size_t o = 0; o < layer->num_output_maps; o++)
	{
		for(size_t j = 0; j < output_size; j++)
		{
			errors[j] *= deltas[j];
			layer->gradients[layer->num_weights - layer->num_output_maps + o] += errors[j];
		}

		for(size_t i = 0; i < layer->num_input_maps; i++)
		{
			correlate_valid(inputs + i * input_size, layer->input_dims, errors, layer->output_dims, gradients);

			gradients += kernel_size;
		}

		errors += output_size;
		deltas += output_size;
	}
}

void conv_update(layer_t *layer, update_rule_t *update_rule)
{
	for(size_t w = 0; w < layer->num_weights; w++)
	{
		layer->weights[w] -= layer->gradients[w] * update_rule->learning_rate;
		layer->gradients[w] = 0.0;
	}
}
