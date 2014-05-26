#include <cstring>
#include <mm_malloc.h>

#include <nnet/types.hpp>

nnet_float *nnet_malloc(size_t length)
{
	return (nnet_float *)_mm_malloc(sizeof(nnet_float) * length, 32);
}

void nnet_free(nnet_float *ptr)
{
	_mm_free(ptr);
}

void nnet_shuffle_instances(nnet_float *features, nnet_float *labels, size_t length, size_t num_features, size_t num_labels)
{
	nnet_float *temp_features = nnet_malloc(num_features);
	nnet_float *temp_labels = nnet_malloc(num_labels);

	for(size_t i = length - 1; i > 0; i--)
	{
		size_t j = rand() % i;

		memcpy(temp_features, features + i * num_features, num_features * sizeof(nnet_float));
		memcpy(features + i * num_features, features + j * num_features, num_features * sizeof(nnet_float));
		memcpy(features + j * num_features, temp_features, num_features * sizeof(nnet_float));

		memcpy(temp_labels, labels + i * num_labels, num_labels * sizeof(nnet_float));
		memcpy(labels + i * num_labels, labels + j * num_labels, num_labels * sizeof(nnet_float));
		memcpy(labels + j * num_labels, temp_labels, num_labels * sizeof(nnet_float));
	}

	nnet_free(temp_features);
	nnet_free(temp_labels);
}
