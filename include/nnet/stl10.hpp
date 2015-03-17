#ifndef _STL10_HPP_
#define _STL10_HPP_

#include <vector>

#include "types.hpp"

size_t loadStl10(const std::vector<const char *> &filenames, std::vector<nnet_float> &features, std::vector<nnet_float> &labels, std::vector<nnet_float> &unlabelled, std::vector<size_t> &indices);

#endif
