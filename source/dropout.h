#ifndef _DROPOUT_H_
#define _DROPOUT_H_

#include "types.h"

typedef struct
{
	nnet_float_t prob;
} dropout_layer_data_t;

layer_t *dropout_create(size_t num_units, nnet_float_t prob);
void dropout_destroy(layer_t *layer);
void dropout_forward(layer_t *layer, nnet_float_t *inputs, int train);
void dropout_backward(layer_t *layer, nnet_float_t *bperrs);

#endif
