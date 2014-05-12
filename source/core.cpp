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
