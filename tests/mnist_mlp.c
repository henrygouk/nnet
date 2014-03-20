#include <stdio.h>
#include <stdlib.h>

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

	layer_t *layers[2];
	layers[0] = full_create(28 * 28, 200, RECTIFIED);
	layers[1] = full_create(200, 10, LOGISTIC);

	update_rule_t *update_rule = (update_rule_t *)malloc(sizeof(update_rule_t));
	update_rule->algorithm = SGD | MOMENTUM;
	update_rule->learning_rate = 0.0005;
	update_rule->momentum_rate = 0.9;

	layers[0]->update_rule = update_rule;
	layers[1]->update_rule = update_rule;

	ffnn_t *ffnn = ffnn_create(layers, 2, SQUARED_ERROR);

	for(size_t i = 0; i < 10; i++)
	{
		ffnn_train(ffnn, features, labels, 60000, 1, 100);

		mnist_evaluate(ffnn, test_features, test_labels);
	}

	ffnn_destroy(ffnn);

	for(size_t i = 0; i < 2; i++)
	{
		layer_destroy(layers[i]);
	}

	free(update_rule);
	nnet_free(features);
	nnet_free(labels);
	nnet_free(test_features);
	nnet_free(test_labels);

	return 0;
}
