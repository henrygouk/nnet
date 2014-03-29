#ifndef _CIFAR10_H_
#define _CIFAR10_H_

nnet_float_t *cifar10_images(const char *filename);
nnet_float_t *cifar10_labels(const char *filename, nnet_float_t neg_value, nnet_float_t pos_value);

#endif
