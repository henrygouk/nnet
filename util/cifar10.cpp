#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nnet/core.hpp>

using namespace std;

/*
    Loads a file containing the cifar10 dataset
*/
void cifar10(const char *filename, nnet_float **images, nnet_float **labels)
{
    FILE *fd = fopen(filename, "rb");
	unsigned char *buffer = (unsigned char *)malloc(60000 * (32 * 32 * 3 + 1));
	unsigned char *buf= buffer;
	nnet_float *imgs = nnet_malloc(60000 * 32 * 32 * 3);
	nnet_float *lbls = nnet_malloc(60000 * 10);
	memset(lbls, 0, sizeof(nnet_float) * 60000 * 10);

	fread(buffer, sizeof(unsigned char), 60000 * (32 * 32 * 3 + 1), fd);

	for(size_t i = 0; i < 60000; i++)
	{
		lbls[i * 10 + *buf] = 1.0;
		buf++;

		for(size_t j = 0; j < 32 * 32 * 3; j++)
		{
			imgs[i * 32 * 32 * 3 + j] = (nnet_float)*buf / 255.0;
			buf++;
		}
	}

	*images = imgs;
	*labels = lbls;

	free(buffer);
	fclose(fd);
}

