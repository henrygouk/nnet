#include <stdlib.h>
#include <string.h>

#include "ffnn.h"

static void forward(ffnn_t *ffnn, nnet_float_t *features, int train);
static void backward(ffnn_t *ffnn, nnet_float_t *features, nnet_float_t *labels);
static void update(ffnn_t *ffnn);
static void calculate_errors(ffnn_t *ffnn, layer_t *layer, nnet_float_t *labels);

ffnn_t *ffnn_create(layer_t **layers, size_t num_layers, loss_function_t loss_function)
{
	if(num_layers == 0)
	{
		return 0;
	}

	ffnn_t *ffnn = (ffnn_t *)malloc(sizeof(ffnn_t));

	ffnn->layers = (layer_t **)malloc(sizeof(layer_t *) * num_layers);
	memcpy(ffnn->layers, layers, num_layers * sizeof(layer_t *));

	ffnn->num_inputs = layers[0]->num_inputs;
	ffnn->num_outputs = layers[num_layers - 1]->num_units;
	ffnn->num_layers = num_layers;
	ffnn->loss_function = loss_function;

	return ffnn;
}

void ffnn_destroy(ffnn_t *ffnn)
{
	free(ffnn->layers);
	free(ffnn);
}

void ffnn_train(ffnn_t *ffnn, nnet_float_t *features, nnet_float_t *labels, size_t num_instances, size_t epochs, size_t batch_size)
{
	for(size_t e = 0; e < epochs; e++)
	{
		for(size_t i = 0; i < num_instances; i++)
		{
			if(i % batch_size == 0)
			{
				for(size_t j = 0; j < ffnn->num_layers; j++)
				{
					layer_start_batch(ffnn->layers[j]);
				}
			}

			forward(ffnn, features + i * ffnn->num_inputs, 1);
			backward(ffnn, features + i * ffnn->num_inputs, labels + i * ffnn->num_outputs);
			
			if((i + 1) % batch_size == 0)
			{
				update(ffnn);
			}
		}
	}
}

void ffnn_predict(ffnn_t *ffnn, nnet_float_t *features, nnet_float_t *labels)
{
	forward(ffnn, features, 0);
	memcpy(labels, ffnn->layers[ffnn->num_layers - 1]->activations, ffnn->layers[ffnn->num_layers - 1]->num_units * sizeof(nnet_float_t));
}

static void forward(ffnn_t *ffnn, nnet_float_t *features, int train)
{
	for(size_t i = 0; i < ffnn->num_layers; i++)
	{
		layer_forward(ffnn->layers[i], features, train);
		features = ffnn->layers[i]->activations;
	}
}

static void backward(ffnn_t *ffnn, nnet_float_t *features, nnet_float_t *labels)
{
	calculate_errors(ffnn, ffnn->layers[ffnn->num_layers - 1], labels);
	
	for(size_t l = ffnn->num_layers - 1; l > 0; l--)
	{
		layer_calculate_gradients(ffnn->layers[l], ffnn->layers[l - 1]->activations);
		layer_backward(ffnn->layers[l], ffnn->layers[l - 1]->errors);
	}

	layer_calculate_gradients(ffnn->layers[0], features);
}

static void update(ffnn_t *ffnn)
{
	for(size_t l = 0; l < ffnn->num_layers; l++)
	{
		layer_end_batch(ffnn->layers[l]);
		layer_update(ffnn->layers[l]);
	}
}

void calculate_errors(ffnn_t *ffnn, layer_t *layer, nnet_float_t *labels)
{
	switch(ffnn->loss_function)
	{
		case SQUARED_ERROR:
			for(size_t u = 0; u < layer->num_units; u++)
			{
				layer->errors[u] = layer->activations[u] - labels[u];
			}

			break;

		default:
			break;
	}
}
