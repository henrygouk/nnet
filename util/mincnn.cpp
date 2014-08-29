#include <iostream>
#include <fstream>
#include <cstdio>
#include <sys/time.h>

#include <nnet/nnet.hpp>

#include "evaluate.hpp"

using namespace std;

int main(int argc, char **argv)
{
	size_t numInstances = 28000;//28423;
	size_t numFeatures = 100 * 100 * 3;
	size_t numLabels = 20;
	
	nnet_float *features = nnet_malloc(numInstances * numFeatures);
	nnet_float *labels = nnet_malloc(numInstances * numLabels);
	
	FILE *fd = fopen(argv[1], "rb");
	fread(features, sizeof(nnet_float), numInstances * numFeatures, fd);
	fclose(fd);

	fd = fopen(argv[2], "rb");
	fread(labels, sizeof(nnet_float), numInstances * numLabels, fd);
	fclose(fd);

	ActivationFunction *rect = new RectifiedLinear();
	ActivationFunction *softmax = new Softmax();
	SGD *ur = new SGD();
	ur->learningRate = 0.0001;
	ur->momentumRate = 0.9;
	ur->l2DecayRate = 0.0001;
	ur->maxNorm = 3.0;

	size_t l1InputDims[] = {100, 100};
	size_t l2InputDims[] = {90, 90};
	size_t l3InputDims[] = {30, 30};
	size_t l4InputDims[] = {24, 24};
	size_t l5InputDims[] = {8, 8};
	size_t kernelDims[] = {11, 11};
	size_t kernelDims2[] = {7, 7};
	size_t poolDims[] = {3, 3};

	vector<Layer *> layers;
	layers.push_back(new Convolutional(2, l1InputDims, kernelDims, 3, 32, 0.0001, rect, ur));
	layers.push_back(new MaxPool(2, l2InputDims, 32, poolDims));
	layers.push_back(new Convolutional(2, l3InputDims, kernelDims2, 32, 32, 0.01, rect, ur));
	layers.push_back(new MaxPool(2, l4InputDims, 32, poolDims));
	layers.push_back(new FullyConnected(32 * l5InputDims[0] * l5InputDims[1], 20, 0.01, softmax, ur));

	FeedForward *ff = new FeedForward(layers, new CrossEntropy());

	cout << "Starting..." << endl;

	for(size_t i = 0; i < 10; i++)
	{
		struct timeval start, end;
		gettimeofday(&start, 0);
		ff->train(features, labels, numInstances, 1, 100);
		gettimeofday(&end, 0);

		float duration = (nnet_float)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0);
		float training = evaluate(ff, features, labels, numInstances, numFeatures, numLabels);

		cout << "Epoch: " << i << "\t";
		cout << "Training time: " << duration << "\t";
		cout << "Training accuracy: " << training << endl;
	}

	ofstream ofs(argv[3]);
	ff->save(ofs);
	ofs.close();

	return 0;
}
