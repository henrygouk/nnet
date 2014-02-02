#ifndef _FFTCONV_H_
#define _FFTCONV_H_

#include "types.h"

layer_t *fftconv_create(size_t num_input_images, size_t input_dims, size_t num_output_images, size_t kernel_dims, activation_function_t func);
void fftconv_destroy(layer_t *layer);
void fftconv_forward(layer_t *layer, nnet_float_t *inputs);
void fftconv_backward(layer_t *layer, nnet_float_t *bperrs);
void fftconv_calculate_gradients(layer_t *layer, nnet_float_t *inputs);
void fftconv_update(layer_t *layer, update_rule_t *update_rule);

#endif
