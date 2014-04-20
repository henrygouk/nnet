#include <stdio.h>
#include <stdlib.h>

#include "../source/core.h"
#include "../source/ffnn.h"
#include "../source/fftconv.h"
#include "../source/full.h"
#include "../source/maxpool.h"
#include "../source/dropout.h"

#include "cifar10.h"
#include "mnist.h"

int main(int argc, char **argv)
{
	nnet_float_t *features;
	nnet_float_t *labels;
	cifar10(argv[1], &features, &labels);

	layer_t *layers[10];
	layers[0] = dropout_create(32 * 32 * 3, 0.8);
	layers[1] = fftconv_create(3, 32, 32, 5, LOGISTIC, 0.1);
	layers[2] = maxpool_create(32, 28, 2);
	layers[3] = dropout_create(32 * 14 * 14, 0.5);
	layers[4] = fftconv_create(32, 14, 32, 5, LOGISTIC, 0.1);
	layers[5] = maxpool_create(32, 10, 2);
	layers[6] = dropout_create(32 * 5 * 5, 0.5);
	layers[7] = full_create(32 * 5 * 5, 100, LOGISTIC, 0.01);
	layers[8] = dropout_create(100, 0.5);
	layers[9] = full_create(100, 10, SOFTMAX, 0.01);

	update_rule_t *update_rule = (update_rule_t *)malloc(sizeof(update_rule_t));
	update_rule->algorithm = SGD | MOMENTUM;
	update_rule->learning_rate = 0.001;
	update_rule->momentum_rate = 0.9;

	update_rule_t *full_update_rule = (update_rule_t *)malloc(sizeof(update_rule_t));
	full_update_rule->algorithm = SGD | MOMENTUM;
	full_update_rule->learning_rate = 0.001;
	full_update_rule->momentum_rate = 0.9;

	layers[1]->update_rule = update_rule;
	layers[4]->update_rule = update_rule;
	layers[7]->update_rule = full_update_rule;
	layers[9]->update_rule = full_update_rule;

	ffnn_t *ffnn = ffnn_create(layers, 10, SQUARED_ERROR);

	printf("Starting CIFAR test...\n");

	for(size_t i = 0; i < 50; i++)
	{
		ffnn_train(ffnn, features, labels, 50000, 1, 100);
		nnet_shuffle_instances(features, labels, 50000, 32 * 32 * 3, 10);

		printf("Epoch: %04lu   Validation: %02.2f%%   Resubstitution: %02.2f%%\n", i + 1, mnist_evaluate(ffnn, features + 32 * 32 * 3 * 50000, labels + 10 * 50000, 10000) * 100.0, mnist_evaluate(ffnn, features, labels, 50000) * 100.0);

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

	return 0;
}
