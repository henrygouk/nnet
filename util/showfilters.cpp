#include <cstdio>
#include <cfloat>
#include <cstring>
#include <iostream>

#include <nnet/nnet.hpp>
#include <CImg.h>

using namespace cimg_library;
using namespace std;

int main(int argc, char **argv)
{
	if(argc != 6)
	{
		cout << "Usage: " << argv[0] << " <nnet file> <filter size> <channels> <number of filters> <output png file>" << endl;
		return 0;
	}

	size_t filterDims = atoi(argv[2]);
	size_t channels = atoi(argv[3]);
	size_t numFilters = atoi(argv[4]);

	FILE *fd = fopen(argv[1], "rb");

	CImg<float> finalImage, row;
	float buf[filterDims * filterDims * channels];

	for(size_t i = 0; i < numFilters; i++)
	{
		CImg<float> img(filterDims, filterDims, 1, channels);

		fread(buf, sizeof(float), filterDims * filterDims * channels, fd);

		float minval = FLT_MAX;
		float maxval = -FLT_MAX;

		for(size_t j = 0; j < filterDims * filterDims * channels; j++)
		{
			if(buf[j] < minval)
			{
				minval = buf[j];
			}
			
			if(buf[j] > maxval)
			{
				maxval = buf[j];
			}
		}

		for(size_t j = 0; j < filterDims * filterDims * channels; j++)
		{
			buf[j] = (buf[j] - minval) / (maxval - minval);
		}

		memcpy(img.data(), buf, sizeof(float) * filterDims * filterDims * channels);

		img.resize(filterDims + 1, filterDims + 1, 1, channels, 0);
		row.append(img);

		if((i + 1) % 16 == 0)
		{
			finalImage.append(row, 'y');
			row.clear();
		}
	}

	if(numFilters % 16 == 0)
	{
		finalImage.append(row, 'y');
	}

	(finalImage * 255.0).resize(finalImage.width() * 5, finalImage.height() * 5, 1, channels).save_png(argv[5]);

	return 0;
}
