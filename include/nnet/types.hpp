#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <cstdint>
#include <functional>

typedef float nnet_float;
typedef std::function<void(const nnet_float *, const nnet_float *, nnet_float *, size_t)> LossFunction;

#endif
