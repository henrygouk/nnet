#include <stdio.h>

#include "../source/core.h"
#include "../source/ffnn.h"
#include "../source/fftconv.h"
#include "../source/full.h"
#include "../source/maxpool.h"

#include "mnist.h"

int main(int argc, char **argv)
{
	nnet_float_t *features = mnist_training_images(argv[1]);
	nnet_float_t *labels = mnist_training_labels(argv[2]);
	nnet_float_t *test_features = mnist_testing_images(argv[3]);
	nnet_float_t *test_labels = mnist_testing_labels(argv[4]);
	
	layer_t *layers[5];
	layers[0] = fftconv_create(1, 28, 16, 5, RECTIFIED);
	layers[1] = maxpool_create(16, 24, 2);
	layers[2] = fftconv_create(16, 12, 16, 5, RECTIFIED);
	layers[3] = maxpool_create(16, 8, 2);
	layers[4] = full_create(16 * 4 * 4, 10, LOGISTIC);
	ffnn_t *ffnn = ffnn_create(layers, 5);
	ffnn->update_rule.learning_rate = 0.002;

	for(size_t i = 0; i < 5; i++)
	{
		ffnn_train(ffnn, features, labels, 60000, 1, 100);

		mnist_evaluate(ffnn, test_features, test_labels);
	}

	ffnn_destroy(ffnn);

	for(size_t i = 0; i < 5; i++)
		layer_destroy(layers[i]);

	nnet_free(features);
	nnet_free(labels);
	nnet_free(test_features);
	nnet_free(test_labels);
}
