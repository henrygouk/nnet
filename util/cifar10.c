#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../source/core.h"
#include "../source/types.h"

/*
    Loads a file containing the cifar10 dataset
*/
void cifar10(const char *filename, nnet_float_t **images, nnet_float_t **labels)
{
    FILE *fd = fopen(filename, "rb");
	unsigned char *buffer = malloc(60000 * (32 * 32 * 3 + 1));
	unsigned char *buf= buffer;
	nnet_float_t *imgs = nnet_malloc(60000 * 32 * 32 * 3);
	nnet_float_t *lbls = nnet_malloc(60000 * 10);
	memset(lbls, 0, sizeof(nnet_float_t) * 60000 * 10);

	fread(buffer, sizeof(unsigned char), 60000 * (32 * 32 * 3 + 1), fd);

	for(size_t i = 0; i < 60000; i++)
	{
		lbls[i * 10 + *buf] = 1.0;
		buf++;

		for(size_t j = 0; j < 32 * 32 * 3; j++)
		{
			imgs[i * 32 * 32 * 3 + j] = (nnet_float_t)*buf / 255.0;
			buf++;
		}
	}

	*images = imgs;
	*labels = lbls;

	free(buffer);
	fclose(fd);
}

