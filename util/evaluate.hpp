#ifndef _EVALUATE_H_
#define _EVALUATE_H_

#include "nnet/nnet.hpp"

nnet_float evaluate(X86FeedForward *ffnn, nnet_float *features, nnet_float *labels, size_t count, size_t num_features, size_t num_outputs);

#endif
