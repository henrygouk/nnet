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
	
	layer_t *layers[6];
	layers[0] = fftconv_create(1, 28, 32, 5, RECTIFIED);
	layers[1] = maxpool_create(32, 24, 2);
	layers[2] = fftconv_create(32, 12, 32, 5, RECTIFIED);
	layers[3] = maxpool_create(32, 8, 2);
	layers[4] = full_create(32 * 4 * 4, 100, RECTIFIED);
	layers[5] = full_create(100, 10, LOGISTIC);

	update_rule_t *update_rule = (update_rule_t *)malloc(sizeof(update_rule_t));
	update_rule->algorithm = SGD | MOMENTUM;
	update_rule->learning_rate = 0.0005;
	update_rule->momentum_rate = 0.9;

	update_rule_t *full_update_rule = (update_rule_t *)malloc(sizeof(update_rule_t));
	full_update_rule->algorithm = SGD | MOMENTUM;
	full_update_rule->learning_rate = 0.0005;
	full_update_rule->momentum_rate = 0.9;

	layers[0]->update_rule = update_rule;
	layers[2]->update_rule = update_rule;
	layers[4]->update_rule = full_update_rule;
	layers[5]->update_rule = full_update_rule;

	ffnn_t *ffnn = ffnn_create(layers, 6, SQUARED_ERROR);

	printf("Starting MNIST test...\n");

	for(size_t i = 0; i < 10; i++)
	{
		ffnn_train(ffnn, features, labels, 60000, 1, 100);

		mnist_evaluate(ffnn, test_features, test_labels);

		update_rule->learning_rate *= 0.9;
		full_update_rule->learning_rate *= 0.9;
	}

	ffnn_destroy(ffnn);

	for(size_t i = 0; i < 6; i++)
	{
		layer_destroy(layers[i]);
	}

	free(update_rule);
	free(full_update_rule);
	nnet_free(features);
	nnet_free(labels);
	nnet_free(test_features);
	nnet_free(test_labels);
}
