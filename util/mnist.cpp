#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <nnet/nnet.hpp>

nnet_float *load_floats(const char *filename, size_t length, size_t offset, int norm)
{
	FILE *fd = fopen(filename, "rb");
	unsigned char *bytedata = new unsigned char[length + offset];
	nnet_float *data = nnet_malloc(length);

	fread(bytedata, length + offset, 1, fd);

	for(size_t i = 0; i < length; i++)
	{
		data[i] = (nnet_float)bytedata[i + offset] * (norm ? (1.0 / 255.0) : 1.0);
	}

	free(bytedata);
	fclose(fd);

	return data;
}

nnet_float *mnist_training_images(const char *filename)
{
	return load_floats(filename, 60000 * 28 * 28, 16, 1);
}

nnet_float *mnist_testing_images(const char *filename)
{
	return load_floats(filename, 10000 * 28 * 28, 16, 1);
}

nnet_float *mnist_training_labels(const char *filename)
{
	nnet_float *ret = nnet_malloc(600000);
	nnet_float *labels = load_floats(filename, 60000, 8, 0);

	memset(ret, 0, sizeof(nnet_float) * 600000);

	for(size_t i = 0; i < 60000; i++)
	{
		ret[i * 10 + (size_t)labels[i]] = 1.0;
	}
	
	nnet_free(labels);

	return ret;
}

nnet_float *mnist_testing_labels(const char *filename)
{
	nnet_float *ret = nnet_malloc(100000);
	nnet_float *labels = load_floats(filename, 10000, 8, 0);

	memset(ret, 0, sizeof(nnet_float) * 100000);

	for(size_t i = 0; i < 10000; i++)
	{
		ret[i * 10 + (size_t)labels[i]] = 1.0;
	}
	
	nnet_free(labels);

	return ret;
}


