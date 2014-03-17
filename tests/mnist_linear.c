#include <stdio.h>
#include <stdlib.h>

#include "../source/conv.h"
#include "../source/core.h"
#include "../source/ffnn.h"
#include "../source/fftconv.h"
#include "../source/full.h"

#include "mnist.h"

int main(int argc, char **argv)
{
	nnet_float_t *features = mnist_training_images(argv[1]);
	nnet_float_t *labels = mnist_training_labels(argv[2]);
	nnet_float_t *test_features = mnist_testing_images(argv[3]);
	nnet_float_t *test_labels = mnist_testing_labels(argv[4]);

	layer_t *layer = full_create(28 * 28, 10, LOGISTIC);
	ffnn_t *ffnn = ffnn_create(&layer, 1);

	layer->update_rule = malloc(sizeof(update_rule_t));
	layer->update_rule->algorithm = SGD;
	layer->update_rule->learning_rate = 0.001;

	ffnn_train(ffnn, features, labels, 60000, 5, 1);

	mnist_evaluate(ffnn, test_features, test_labels);

	ffnn_destroy(ffnn);
	layer_destroy(layer);

	nnet_free(features);
	nnet_free(labels);
	nnet_free(test_features);
	nnet_free(test_labels);
}


