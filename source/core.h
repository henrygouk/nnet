#ifndef _CORE_H_
#define _CORE_H_

#include "types.h"

nnet_float_t *nnet_malloc(size_t length);
void nnet_free(nnet_float_t *ptr);
void random_vector(nnet_float_t *weight, size_t length, nnet_float_t lower, nnet_float_t upper);
void nnet_shuffle_instances(nnet_float_t *features, nnet_float_t *labels, size_t length, size_t num_features, size_t num_labels);

#endif
