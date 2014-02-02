#include "../source/core.h"
#include "../source/types.h"

nnet_float_t *load_floats(const char *filename, size_t length, size_t offset)
{
	FILE *fd = fopen(filename, "rb");
	unsigned char *bytedata = malloc(length + offset);
	nnet_float_t *data = nnet_malloc(length);

	for(size_t i = 0; i < length; i++)
	{
		data[i] = (nnet_float_t)bytedata[i + 16];
	}

	free(bytedata);
	fclose(fd);

	return data;
}

nnet_float_t *mnist_training_images(const char *filename)
{
	return load_floats(filename, 60000 * 28 * 28, 16);
}


