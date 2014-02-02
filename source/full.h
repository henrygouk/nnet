#ifndef _FULL_H_
#define _FULL_H_

#include "types.h"

layer_t *full_create(size_t num_inputs, size_t num_units, activation_function_t func);
void full_destroy(layer_t *layer);
void full_forward(layer_t *layer, nnet_float_t *inputs);
void full_backward(layer_t *layer, nnet_float_t *bperrs);
void full_calculate_gradients(layer_t *layer, nnet_float_t *inputs);
void full_update(layer_t *layer, update_rule_t *update_rule);

#endif
