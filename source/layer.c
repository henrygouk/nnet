#include <float.h>

#include "types.h"
#include "update.h"

void layer_destroy(layer_t *layer)
{
	layer->vtable->destroy(layer);
}

void layer_forward(layer_t *layer, nnet_float_t *features, int train)
{
	layer->vtable->forward(layer, features, train);
}

void layer_calculate_activations(layer_t *layer, nnet_float_t *delta_activations, activation_function_t activation_function)
{
	switch(activation_function)
	{
		case NONE:
		{
			for(size_t i = 0; i < layer->num_units; i++)
			{
				delta_activations[i] = 1.0;
			}

			break;
		}

		case RECTIFIED:
		{
			for(size_t i = 0; i < layer->num_units; i++)
			{
				if(layer->activations[i] < 0)
				{
					delta_activations[i] = 0.0;
					layer->activations[i] = 0.0;
				}
				else
				{
					delta_activations[i] = 1.0;
				}
			}

			break;
		}

		case LOGISTIC:
		{
			for(size_t i = 0; i < layer->num_units; i++)
			{
				layer->activations[i] = 1.0 / (1.0 + EXP(-layer->activations[i]));
				delta_activations[i] = layer->activations[i] * (1.0 - layer->activations[i]);
			}

			break;
		}

		case SOFTMAX:
		{
			nnet_float_t maxval = -FLT_MAX;
			nnet_float_t sum = 0.0;

			for(size_t i = 0; i < layer->num_units; i++)
			{
				if(layer->activations[i] > maxval)
				{
					maxval = layer->activations[i];
				}
			}

			for(size_t i = 0; i < layer->num_units; i++)
			{
				layer->activations[i] = EXP(layer->activations[i] - maxval);
				sum += layer->activations[i];
			}

			sum = 1.0 / sum;

			for(size_t i = 0; i < layer->num_units; i++)
			{
				layer->activations[i] *= sum;
				delta_activations[i] = layer->activations[i] * (1.0 - layer->activations[i]);
			}

			break;
		}
	}
}

void layer_backward(layer_t *layer, nnet_float_t *bperrs)
{
	layer->vtable->backward(layer, bperrs);
}

void layer_calculate_gradients(layer_t *layer, nnet_float_t *features)
{
	if(layer->vtable->calculate_gradients)
		layer->vtable->calculate_gradients(layer, features);
}

void layer_start_batch(layer_t *layer)
{
	if(layer->vtable->start_batch)
		layer->vtable->start_batch(layer);
}

void layer_end_batch(layer_t *layer)
{
	if(layer->vtable->end_batch)
		layer->vtable->end_batch(layer);
}

void layer_update(layer_t *layer)
{
	if(!layer->update_rule)
		return;

	switch(layer->update_rule->algorithm)
	{
		case SGD:
			update_sgd(layer);
			break;

		case (SGD | MOMENTUM):
			update_sgd_momentum(layer);
			break;

		case (SGD | MOMENTUM | L2_DECAY):
			update_sgd_momentum_l2_decay(layer);
			break;

		default:
			break;
	}
}
