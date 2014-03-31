#include <stdlib.h>
#include <string.h>

#include "conv.h"
#include "core.h"
#include "vector.h"

layer_vtable_t conv_vtable;

layer_t *conv_create(size_t num_input_images, size_t input_dims, size_t num_output_images, size_t kernel_dims, activation_function_t func, nnet_float_t weight_size)
{
	layer_t *layer = (layer_t *)malloc(sizeof(layer_t));
	conv_layer_data_t *layer_data = (conv_layer_data_t *)malloc(sizeof(conv_layer_data_t));

	size_t output_dims = input_dims - kernel_dims + 1;
	layer->num_inputs = num_input_images * input_dims * input_dims;
	layer->num_units = num_output_images * output_dims * output_dims;
	layer->num_weights = num_input_images * num_output_images * kernel_dims * kernel_dims + num_output_images;
	layer_data->padded = nnet_malloc(kernel_dims * kernel_dims);
	layer->weights = nnet_malloc(layer->num_weights);
	layer->gradients = nnet_malloc(layer->num_weights);
	layer->activations = nnet_malloc(layer->num_units);
	layer->errors = nnet_malloc(layer->num_units);
	layer_data->delta_activations = nnet_malloc(layer->num_units);
	layer_data->input_dims = input_dims;
	layer_data->output_dims = output_dims;
	layer_data->kernel_dims = kernel_dims;
	layer_data->num_input_maps = num_input_images;
	layer_data->num_output_maps = num_output_images;
	layer_data->activation_function = func;

	random_vector(layer->weights, layer->num_weights, -weight_size, weight_size);
	memset(layer->gradients, 0, sizeof(nnet_float_t) * layer->num_weights);

	conv_vtable.destroy = &conv_destroy;
	conv_vtable.forward = &conv_forward;
	conv_vtable.backward = &conv_backward;
	conv_vtable.calculate_gradients = &conv_calculate_gradients;
	conv_vtable.start_batch = 0;
	conv_vtable.end_batch = 0;

	layer->vtable = &conv_vtable;
	layer->layer_data = layer_data;

	return layer;
}

void conv_destroy(layer_t *layer)
{
	conv_layer_data_t *layer_data = layer->layer_data;

	nnet_free(layer->weights);
	nnet_free(layer->gradients);
	nnet_free(layer->activations);
	nnet_free(layer->errors);
	nnet_free(layer_data->delta_activations);
	nnet_free(layer_data->padded);
	free(layer->layer_data);
	free(layer);
}

void conv_forward(layer_t *layer, nnet_float_t *inputs, int train)
{
	conv_layer_data_t *layer_data = (conv_layer_data_t *)layer->layer_data;
	nnet_float_t *output = layer->activations;
	nnet_float_t *weights = layer->weights;

	//Iterate over each output map
	for(size_t u = 0; u < layer_data->num_output_maps; u++)
	{
		//Set all pixels for this output map to the bias value
		for(size_t p = 0; p < layer_data->output_dims * layer_data->output_dims; p++)
		{
			output[p] = layer->weights[layer->num_weights - layer_data->num_output_maps + u];
		}

		nnet_float_t *input = inputs;

		//iterate over each input map
		for(size_t i = 0; i < layer_data->num_input_maps; i++)
		{
			convolve_valid(input, layer_data->input_dims, weights, layer_data->kernel_dims, output);

			input += layer_data->input_dims * layer_data->input_dims;
			weights += layer_data->kernel_dims * layer_data->kernel_dims;
		}
		
		output += layer_data->output_dims * layer_data->output_dims;
	}

	layer_calculate_activations(layer, layer_data->delta_activations, layer_data->activation_function);
}

void conv_backward(layer_t *layer, nnet_float_t *bperrs)
{
	conv_layer_data_t *layer_data = (conv_layer_data_t *)layer->layer_data;
	const size_t kernel_size = layer_data->kernel_dims * layer_data->kernel_dims;
	const size_t input_size = layer_data->input_dims * layer_data->input_dims;
	const size_t output_size = layer_data->output_dims * layer_data->output_dims;

	for(size_t i = 0; i < layer_data->num_input_maps; i++)
	{
		nnet_float_t *kernels = layer->weights + i * kernel_size;
		memset(bperrs, 0, input_size * sizeof(nnet_float_t));

		for(size_t o = 0; o < layer_data->num_output_maps; o++)
		{
			correlate_full(layer->errors + o * output_size, layer_data->output_dims, kernels, layer_data->kernel_dims, bperrs);

			kernels += kernel_size * layer_data->num_input_maps;
		}

		bperrs += input_size;
	}
}

void conv_calculate_gradients(layer_t *layer, nnet_float_t *inputs)
{
	conv_layer_data_t *layer_data = (conv_layer_data_t *)layer->layer_data;
	const size_t kernel_size = layer_data->kernel_dims * layer_data->kernel_dims;
	const size_t input_size = layer_data->input_dims * layer_data->input_dims;
	const size_t output_size = layer_data->output_dims * layer_data->output_dims;
	nnet_float_t *errors = layer->errors;
	nnet_float_t *deltas = layer_data->delta_activations;
	nnet_float_t *gradients = layer->gradients;

	for(size_t o = 0; o < layer_data->num_output_maps; o++)
	{
		for(size_t j = 0; j < output_size; j++)
		{
			errors[j] *= deltas[j];
			layer->gradients[layer->num_weights - layer_data->num_output_maps + o] += errors[j];
		}

		for(size_t i = 0; i < layer_data->num_input_maps; i++)
		{
			memset(layer_data->padded, 0, sizeof(nnet_float_t) * kernel_size);
			correlate_valid(inputs + i * input_size, layer_data->input_dims, errors, layer_data->output_dims, layer_data->padded);
			rotate_180(layer_data->padded, layer_data->kernel_dims, gradients);

			gradients += kernel_size;
		}

		errors += output_size;
		deltas += output_size;
	}
}
