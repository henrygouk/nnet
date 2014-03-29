#include <mm_malloc.h>
#include <string.h>

#include "core.h"

nnet_float_t *nnet_malloc(size_t length)
{
	return _mm_malloc(sizeof(nnet_float_t) * length, 32);
}

void nnet_free(nnet_float_t *ptr)
{
	_mm_free(ptr);
}

void nnet_shuffle_instances(nnet_float_t *features, nnet_float_t *labels, size_t length, size_t num_features, size_t num_labels)
{
	nnet_float_t *temp_features = nnet_malloc(num_features);
	nnet_float_t *temp_labels = nnet_malloc(num_labels);

	for(size_t i = length - 1; i > 0; i--)
	{
		size_t j = rand() % i;
		
		memcpy(temp_features, features + i * num_features, num_features * sizeof(nnet_float_t));
		memcpy(features + i * num_features, features + j * num_features, num_features * sizeof(nnet_float_t));
		memcpy(features + j * num_features, temp_features, num_features * sizeof(nnet_float_t));

		memcpy(temp_labels, labels + i * num_labels, num_labels * sizeof(nnet_float_t));
		memcpy(labels + i * num_labels, labels + j * num_labels, num_labels * sizeof(nnet_float_t));
		memcpy(labels + j * num_labels, temp_labels, num_labels * sizeof(nnet_float_t));
	}

	nnet_free(temp_features);
	nnet_free(temp_labels);
}

void random_vector(nnet_float_t *vector, size_t length, nnet_float_t lower, nnet_float_t upper)
{
	for(size_t i = 0; i < length; i++)
	{
		vector[i] = (rand() / (nnet_float_t)RAND_MAX) * (upper - lower) + lower;
	}
}
