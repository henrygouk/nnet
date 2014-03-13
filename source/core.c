#include <mm_malloc.h>

#include "core.h"

nnet_float_t *nnet_malloc(size_t length)
{
	return _mm_malloc(sizeof(nnet_float_t) * length, 32);
}

void nnet_free(nnet_float_t *ptr)
{
	_mm_free(ptr);
}

void random_vector(nnet_float_t *vector, size_t length, nnet_float_t lower, nnet_float_t upper)
{
	for(size_t i = 0; i < length; i++)
	{
		vector[i] = (rand() / (nnet_float_t)RAND_MAX) * (upper - lower) + lower;
	}
}
