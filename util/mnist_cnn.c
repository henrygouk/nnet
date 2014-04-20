#include <stdio.h>
#include <stdlib.h>

#include "../source/core.h"
#include "../source/dropout.h"
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
	
	layer_t *layers[10];
	layers[0] = dropout_create(28 * 28, 0.8);
	layers[1] = fftconv_create(1, 28, 32, 5, RECTIFIED, 0.02);
	layers[2] = maxpool_create(32, 24, 2);
	layers[3] = dropout_create(32 * 12 * 12, 0.5);
	layers[4] = fftconv_create(32, 12, 32, 5, RECTIFIED, 0.02);
	layers[5] = maxpool_create(32, 8, 2);
	layers[6] = dropout_create(32 * 4 * 4, 0.5);
	layers[7] = full_create(32 * 4 * 4, 100, RECTIFIED, 0.02);
	layers[8] = dropout_create(100, 0.5);
	layers[9] = full_create(100, 10, SOFTMAX, 0.02);

	update_rule_t *update_rule = (update_rule_t *)malloc(sizeof(update_rule_t));
	update_rule->algorithm = SGD | MOMENTUM;
	update_rule->learning_rate = 0.0001;
	update_rule->momentum_rate = 0.9;

	update_rule_t *full_update_rule = (update_rule_t *)malloc(sizeof(update_rule_t));
	full_update_rule->algorithm = SGD | MOMENTUM;
	full_update_rule->learning_rate = 0.0001;
	full_update_rule->momentum_rate = 0.9;

	layers[1]->update_rule = update_rule;
	layers[4]->update_rule = update_rule;
	layers[7]->update_rule = full_update_rule;
	layers[9]->update_rule = full_update_rule;

	ffnn_t *ffnn = ffnn_create(layers, 10, SQUARED_ERROR);

	printf("Starting MNIST test...\n");

	for(size_t i = 0; i < 50; i++)
	{
		ffnn_train(ffnn, features, labels, 60000, 1, 100);
		nnet_shuffle_instances(features, labels, 60000, 28 * 28, 10);

		//printf("Epoch: %04lu   Validation: %02.2f%%   Resubstitution: %02.2f%%\n", i + 1, mnist_evaluate(ffnn, test_features, test_labels, 10000) * 100.0, mnist_evaluate(ffnn, features, labels, 60000) * 100.0);

		update_rule->learning_rate *= 0.95;
		full_update_rule->learning_rate *= 0.95;
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
