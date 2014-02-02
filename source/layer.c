#include "types.h"

void layer_destroy(layer_t *layer)
{
	layer->vtable->destroy(layer);
}

void layer_forward(layer_t *layer, nnet_float_t *features)
{
	layer->vtable->forward(layer, features);
}

void layer_calculate_activations(layer_t *layer)
{
	switch(layer->activation_function)
	{
		case NONE:
		{
			for(size_t i = 0; i < layer->num_units; i++)
			{
				layer->delta_activations[i] = 1.0;
			}

			break;
		}

		case RECTIFIED:
		{
			for(size_t i = 0; i < layer->num_units; i++)
			{
				if(layer->activations[i] < 0)
				{
					layer->delta_activations[i] = 0.0;
					layer->activations[i] = 0.0;
				}
				else
				{
					layer->delta_activations[i] = 1.0;
				}
			}

			break;
		}

		case LOGISTIC:
		{
			for(size_t i = 0; i < layer->num_units; i++)
			{
				layer->activations[i] = 1.0 / (1.0 + EXP(-layer->activations[i]));
				layer->delta_activations[i] = layer->activations[i] * (1.0 - layer->activations[i]);
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

void layer_update(layer_t *layer, update_rule_t *update_rule)
{
	if(layer->vtable->update)
		layer->vtable->update(layer, update_rule);
}
