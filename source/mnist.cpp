#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "nnet/types.hpp"
#include "nnet/core.hpp"

using namespace std;

nnet_float *load_floats(const char *filename, size_t length, size_t offset, int norm)
{
	FILE *fd = fopen(filename, "rb");
	unsigned char *bytedata = new unsigned char[length + offset];
	nnet_float *data = nnet_malloc(length);

	size_t ret = fread(bytedata, length + offset, 1, fd);

	if(ret < length + offset)
	{
		cerr << "load_floats() Warning: read " << ret << " bytes, expected " << length + offset << endl;
	}

	for(size_t i = 0; i < length; i++)
	{
		data[i] = (nnet_float)bytedata[i + offset] * (norm ? (1.0 / 255.0) : 1.0);
	}

	free(bytedata);
	fclose(fd);

	return data;
}

nnet_float *mnistTrainingImages(const char *filename)
{
	return load_floats(filename, 60000 * 28 * 28, 16, 1);
}

nnet_float *mnistTestingImages(const char *filename)
{
	return load_floats(filename, 10000 * 28 * 28, 16, 1);
}

nnet_float *mnistTrainingLabels(const char *filename)
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

nnet_float *mnistTestingLabels(const char *filename)
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

size_t loadMnist(const vector<const char *> &filenames, vector<nnet_float> &features, vector<nnet_float> &labels)
{
	nnet_float *trainImages = mnistTrainingImages(filenames[0]);
	nnet_float *testImages = mnistTestingImages(filenames[1]);
	nnet_float *trainLabels = mnistTrainingLabels(filenames[2]);
	nnet_float *testLabels = mnistTestingLabels(filenames[3]);

	features.insert(features.end(), trainImages, trainImages + 28 * 28 * 60000);
	features.insert(features.end(), testImages, testImages + 28 * 28 * 10000);
	labels.insert(labels.end(), trainLabels, trainLabels + 10 * 60000);
	labels.insert(labels.end(), testLabels, testLabels + 10 * 10000);

	nnet_free(trainImages);
	nnet_free(testImages);
	nnet_free(trainLabels);
	nnet_free(testLabels);

	//There are 70,000 instances in total (training + test)
	return 70000;
}
