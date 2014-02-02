#ifndef _FFNN_H_
#define _FFNN_H_

#include "types.h"

typedef struct
{
	layer_t **layers;
	size_t num_layers;
	size_t num_inputs;
	size_t num_outputs;
	update_rule_t update_rule;
} ffnn_t;

ffnn_t *ffnn_create(layer_t **layers, size_t num_layers);
void ffnn_destroy(ffnn_t *ffnn);
void ffnn_train(ffnn_t *ffnn, nnet_float_t *features, nnet_float_t *labels, size_t num_instances, size_t epochs, size_t batch_size);
void ffnn_predict(ffnn_t *ffnn, nnet_float_t *feautres, nnet_float_t *labels);

#endif
