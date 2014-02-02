#include <stdio.h>

#include "../source/conv.h"
#include "../source/core.h"
#include "../source/ffnn.h"
#include "../source/fftconv.h"
#include "../source/full.h"

int main(int argc, char **argv)
{
	nnet_float_t *features = nnet_malloc(10);
	nnet_float_t *labels = nnet_malloc(10);

	for(size_t i = 0; i < 10; i++)
	{
		features[i] = i;
		labels[i] = (nnet_float_t)i * 2.0 - 5.0;
	}

	layer_t *layer = full_create(1, 1, NONE);
	ffnn_t *ffnn = ffnn_create(&layer, 1);

	ffnn_train(ffnn, features, labels, 10, 10000, 10);

	printf("%f\t%f\n", layer->weights[0], layer->weights[1]);

	ffnn_destroy(ffnn);
	layer_destroy(layer);

	nnet_free(features);
	nnet_free(labels);
}
