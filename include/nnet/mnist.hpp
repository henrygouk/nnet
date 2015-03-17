#ifndef _MNIST_H_
#define _MNIST_H_

#include <vector>

size_t loadMnist(const std::vector<const char *> &filenames, std::vector<nnet_float> &features, std::vector<nnet_float> &labels);

#endif
