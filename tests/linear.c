#include <stdio.h>

#include "../source/conv.h"
#include "../source/core.h"
#include "../source/ffnn.h"
#include "../source/fftconv.h"
#include "../source/full.h"

#include "mnist.h"

void evaluate(ffnn_t *ffnn, nnet_float_t *features, nnet_float_t *labels);

int main(int argc, char **argv)
{
	nnet_float_t *features = mnist_training_images(argv[1]);
	nnet_float_t *labels = mnist_training_labels(argv[2]);
	nnet_float_t *test_features = mnist_testing_images(argv[3]);
	nnet_float_t *test_labels = mnist_testing_labels(argv[4]);

	layer_t *layer = fftconv_create(1, 28, 10, 28, LOGISTIC);//full_create(28 * 28, 10, LOGISTIC);
	ffnn_t *ffnn = ffnn_create(&layer, 1);
	ffnn->update_rule.learning_rate = 0.001;

	ffnn_train(ffnn, features, labels, 60000, 1, 1);

	evaluate(ffnn, test_features, test_labels);

	ffnn_destroy(ffnn);
	layer_destroy(layer);

	nnet_free(features);
	nnet_free(labels);
	nnet_free(test_features);
	nnet_free(test_labels);
}

void evaluate(ffnn_t *ffnn, nnet_float_t *features, nnet_float_t *labels)
{
	nnet_float_t *output = nnet_malloc(10);
	size_t correct = 0;

	for(size_t i = 0; i < 10000; i++)
	{
		ffnn_predict(ffnn, features + i * 28 * 28, output);

		size_t output_maxind = 0;
		nnet_float_t output_maxval = output[0];
		size_t labels_maxind = 0;
		nnet_float_t labels_maxval = labels[i * 10];

		for(size_t j = 1; j < 10; j++)
		{
			if(output[j] > output_maxval)
			{
				output_maxind = j;
				output_maxval = output[j];
			}

			if(labels[i * 10 + j] > labels_maxval)
			{
				labels_maxind = j;
				labels_maxval = labels[i * 10 + j];
			}
		}

		if(output_maxind == labels_maxind)
			correct++;
	}

	printf("%lu/10000 (%f%%) correct\n", correct, (float)correct/100.0);

	nnet_free(output);
}
