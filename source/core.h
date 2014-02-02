#ifndef _CORE_H_
#define _CORE_H_

#include "types.h"

nnet_float_t *nnet_malloc(size_t length);
void nnet_free(nnet_float_t *ptr);
void random_vector(nnet_float_t *weight, size_t length, nnet_float_t lower, nnet_float_t upper);

#endif
