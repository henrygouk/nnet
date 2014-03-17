#include "update.h"

void update_sgd(layer_t *layer)
{
	for(size_t w = 0; w < layer->num_weights; w++)
	{
		layer->weights[w] -= layer->gradients[w] * layer->update_rule->learning_rate;
		layer->gradients[w] = 0.0;
	}
}

void update_sgd_momentum(layer_t *layer)
{
	for(size_t w = 0; w < layer->num_weights; w++)
	{
		layer->weights[w] -= layer->gradients[w] * layer->update_rule->learning_rate;
		layer->gradients[w] *= layer->update_rule->momentum_rate;
	}
}
