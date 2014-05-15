#ifndef _CORE_HPP_
#define _CORE_HPP_

#include "types.hpp"

nnet_float *nnet_malloc(std::size_t length);
void nnet_free(nnet_float *ptr);
void nnet_shuffle_instances(nnet_float *features, nnet_float *labels, size_t length, size_t num_features, size_t num_labels);

#endif
