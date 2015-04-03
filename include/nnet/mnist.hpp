#ifndef _MNIST_HPP_
#define _MNIST_HPP_

#include <vector>

size_t loadMnist(const std::vector<const char *> &filenames, std::vector<nnet_float> &features, std::vector<nnet_float> &labels);

#endif
