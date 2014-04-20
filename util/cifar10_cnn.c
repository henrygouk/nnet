#include <stdio.h>
#include <stdlib.h>

#include "../source/core.h"
#include "../source/ffnn.h"
#include "../source/fftconv.h"
#include "../source/full.h"
#include "../source/maxpool.h"

#include "cifar10.h"
#include "mnist.h"

int main(int argc, char **argv)
{
	nnet_float_t *features = cifar10_images(argv[1]);
	nnet_float_t *labels = cifar10_labels(argv[2], 1.0, 0.0);

	layer_t *layers[6];
	layers[0] = fftconv_create(3, 32, 32, 5, RECTIFIED, 0.1);
	layers[1] = maxpool_create(32, 28, 2);
	layers[2] = fftconv_create(32, 14, 32, 5, RECTIFIED, 0.1);
	layers[3] = maxpool_create(32, 10, 2);
	layers[4] = full_create(32 * 5 * 5, 100, RECTIFIED, 0.1);
	layers[5] = full_create(100, 10, LOGISTIC, 0.1);

	update_rule_t *update_rule = (update_rule_t *)malloc(sizeof(update_rule_t));
	update_rule->algorithm = SGD | MOMENTUM | L2_DECAY;
	update_rule->learning_rate = 0.001;
	update_rule->momentum_rate = 0.9;
	update_rule->l2_decay_rate = 0.004;

	update_rule_t *full_update_rule = (update_rule_t *)malloc(sizeof(update_rule_t));
	full_update_rule->algorithm = SGD | MOMENTUM | L2_DECAY;
	full_update_rule->learning_rate = 0.001;
	full_update_rule->momentum_rate = 0.9;
	full_update_rule->l2_decay_rate = 0.004;

	layers[0]->update_rule = update_rule;
	layers[2]->update_rule = update_rule;
	layers[4]->update_rule = full_update_rule;
	layers[5]->update_rule = full_update_rule;

	ffnn_t *ffnn = ffnn_create(layers, 6, SQUARED_ERROR);

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
