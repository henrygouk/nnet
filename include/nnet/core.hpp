#ifndef _CORE_HPP_
#define _CORE_HPP_

#include "types.hpp"

nnet_float *nnet_malloc(std::size_t length);
void nnet_free(nnet_float *ptr);

#endif
