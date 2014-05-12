#ifndef _LOSS_HPP_
#define _LOSS_HPP_

#include <nnet/types.hpp>

void squaredError(const nnet_float *h, const nnet_float *l, nnet_float *d, std::size_t len);
void crossEntropy(const nnet_float *h, const nnet_float *l, nnet_float *d, std::size_t len);

#endif
