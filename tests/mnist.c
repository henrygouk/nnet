#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "../source/core.h"
#include "../source/ffnn.h"
#include "../source/types.h"

nnet_float_t *load_floats(const char *filename, size_t length, size_t offset, int norm)
{
	FILE *fd = fopen(filename, "rb");
	unsigned char *bytedata = malloc(length + offset);
	nnet_float_t *data = nnet_malloc(length);

	fread(bytedata, length + offset, 1, fd);

	for(size_t i = 0; i < length; i++)
	{
		data[i] = (nnet_float_t)bytedata[i + offset] * (norm ? (1.0 / 255.0) : 1.0);
	}

	free(bytedata);
	fclose(fd);

	return data;
}

nnet_float_t *mnist_training_images(const char *filename)
{
	return load_floats(filename, 60000 * 28 * 28, 16, 1);
}

nnet_float_t *mnist_testing_images(const char *filename)
{
	return load_floats(filename, 10000 * 28 * 28, 16, 1);
}

nnet_float_t *mnist_training_labels(const char *filename)
{
	nnet_float_t *ret = nnet_malloc(600000);
	nnet_float_t *labels = load_floats(filename, 60000, 8, 0);

	memset(ret, 0, sizeof(nnet_float_t) * 600000);

	for(size_t i = 0; i < 60000; i++)
	{
		ret[i * 10 + (size_t)labels[i]] = 1.0;
	}
	
	nnet_free(labels);

	return ret;
}

nnet_float_t *mnist_testing_labels(const char *filename)
{
	nnet_float_t *ret = nnet_malloc(100000);
	nnet_float_t *labels = load_floats(filename, 10000, 8, 0);

	memset(ret, 0, sizeof(nnet_float_t) * 100000);

	for(size_t i = 0; i < 10000; i++)
	{
		ret[i * 10 + (size_t)labels[i]] = 1.0;
	}
	
	nnet_free(labels);

	return ret;
}

void mnist_evaluate(ffnn_t *ffnn, nnet_float_t *features, nnet_float_t *labels)
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
