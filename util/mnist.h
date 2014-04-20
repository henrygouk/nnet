#ifndef _MNIST_H_
#define _MNIST_H_

nnet_float_t *mnist_training_images(const char *filename);
nnet_float_t *mnist_testing_images(const char *filename);
nnet_float_t *mnist_training_labels(const char *filename);
nnet_float_t *mnist_testing_labels(const char *filename);

#endif
