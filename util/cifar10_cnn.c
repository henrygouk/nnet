#include <stdio.h>
#include <stdlib.h>

#include "../source/core.h"
#include "../source/ffnn.h"
#include "../source/fftconv.h"
#include "../source/full.h"
#include "../source/maxpool.h"
#include "../source/dropout.h"

#include "cifar10.h"
#include "evaluate.h"

int main(int argc, char **argv)
{
	nnet_float_t *features;
	nnet_float_t *labels;
	cifar10(argv[1], &features, &labels);

	layer_t *layers[9];
	layers[0] = dropout_create(32 * 32 * 3, 0.8);
	layers[1] = fftconv_create(3, 32, 64, 5, RECTIFIED, 0.02);
	layers[2] = maxpool_create(64, 28, 2);
	layers[3] = fftconv_create(64, 14, 64, 5, RECTIFIED, 0.02);
	layers[4] = maxpool_create(64, 10, 2);
	layers[5] = dropout_create(64 * 5 * 5, 0.5);
	layers[6] = full_create(64 * 5 * 5, 200, RECTIFIED, 0.08);
	layers[7] = dropout_create(200, 0.5);
	layers[8] = full_create(100, 10, SOFTMAX, 0.08);

	update_rule_t *update_rule = (update_rule_t *)malloc(sizeof(update_rule_t));
	update_rule->algorithm = SGD | MOMENTUM;
	update_rule->learning_rate = 0.001;
	update_rule->momentum_rate = 0.9;

	update_rule_t *full_update_rule = (update_rule_t *)malloc(sizeof(update_rule_t));
	full_update_rule->algorithm = SGD | MOMENTUM;
	full_update_rule->learning_rate = 0.001;
	full_update_rule->momentum_rate = 0.9;

	layers[1]->update_rule = update_rule;
	layers[3]->update_rule = update_rule;
	layers[6]->update_rule = full_update_rule;
	layers[8]->update_rule = full_update_rule;

	ffnn_t *ffnn = ffnn_create(layers, 9, SQUARED_ERROR);

	printf("Starting CIFAR test...\n");

	for(size_t i = 0; i < 50; i++)
	{
		nnet_shuffle_instances(features, labels, 50000, 32 * 32 * 3, 10);
		ffnn_train(ffnn, features, labels, 50000, 1, 100);

		nnet_float_t valid_eval = evaluate(ffnn, features + 32 * 32 * 3 * 50000, labels + 10 * 50000, 10000, 32 * 32 * 3, 10) * 100.0;
		nnet_float_t resub_eval = evaluate(ffnn, features, labels, 50000, 32 * 32 * 3, 10) * 100.0;

		printf("Epoch: %04lu   Validation: %02.2f%%   Resubstitution: %02.2f%%\n", i + 1, valid_eval, resub_eval);
	}

	ffnn_destroy(ffnn);

	for(size_t i = 0; i < 9; i++)
	{
		layer_destroy(layers[i]);
	}

	free(update_rule);
	free(full_update_rule);
	nnet_free(features);
	nnet_free(labels);

	return 0;
}
