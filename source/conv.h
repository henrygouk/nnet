#ifndef _CONV_H_
#define _CONV_H_

#include "types.h"

layer_t *conv_create(size_t num_input_images, size_t input_dims, size_t num_output_images, size_t kernel_dims, activation_function_t func);
void conv_destroy(layer_t *layer);
void conv_forward(layer_t *layer, nnet_float_t *inputs);
void conv_backward(layer_t *layer, nnet_float_t *bperrs);
void conv_calculate_gradients(layer_t *layer, nnet_float_t *inputs);
void conv_update(layer_t *layer, update_rule_t *update_rule);

#endif
