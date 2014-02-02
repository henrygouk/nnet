#ifndef _MAXPOOL_H_
#define _MAXPOOL_H_

layer_t *maxpool_create(size_t num_input_images, size_t input_dims, size_t pool_dims);
void maxpool_destroy(layer_t *layer);
void maxpool_forward(layer_t *layer, nnet_float_t *inputs);
void maxpool_backward(layer_t *layer, nnet_float_t *bperrs);

#endif
