#include <CImg.h>
#include <nnet/nnet.hpp>

#include <string>
#include <iostream>
#include <fstream>

#include "evaluate.hpp"

using namespace std;
using namespace cimg_library;

void createReflections(const nnet_float *input, nnet_float *output, size_t num);

nnet_float *corelFeatures(string filename)
{
	nnet_float *features = nnet_malloc(10000 * 128 * 128 * 3);

	FILE *fp = fopen(filename.c_str(), "rb");
	fread(features, sizeof(nnet_float), 128 * 128 * 3 * 10000, fp);
	fclose(fp);

	return features;
}

nnet_float *corelLabels(size_t numLabels)
{
	nnet_float *labels = nnet_malloc(10000 * numLabels);
	memset(labels, 0, sizeof(nnet_float) * 10000 * numLabels);

	for(size_t i = 0; i < 10000; i++)
	{
		labels[i * numLabels + (i / 100)] = 1.0;
	}

	return labels;
}

int main(int argc, char **argv)
{
	size_t numFeatures = 128 * 128 * 3;
	size_t numLabels = 100;
	size_t numTraining = 9000;
	size_t numValidation = 1000;
	nnet_float *features = corelFeatures(string(argv[1]));
	nnet_float *labels = corelLabels(numLabels);

	Loss *loss = new CrossEntropy();
	ActivationFunction *rect = new RectifiedLinear();
	ActivationFunction *sm = new Softmax();
	SGD *ur = new SGD();
	ur->learningRate = 0.0001;
	ur->momentumRate = 0.9;
	ur->l2DecayRate = 0.0001;
	ur->maxNorm = 3.0;

	size_t l1InputDims[] = {128, 128};
	size_t l2InputDims[] = {120, 120};
	size_t l3InputDims[] = {30, 30};
	size_t l4InputDims[] = {24, 24};
	size_t l5InputDims[] = {6, 6};
	size_t l1KernelDims[] = {9, 9};
	size_t l2PoolDims[] = {4, 4};
	size_t l3KernelDims[] = {7, 7};
	size_t l4PoolDims[] = {4, 4};

	vector<Layer *> layers;
	layers.push_back(new Convolutional(2, l1InputDims, l1KernelDims, 3, 32, 0.0001, rect, ur));
	layers.push_back(new MaxPool(2, l2InputDims, 32, l2PoolDims));
	layers.push_back(new Convolutional(2, l3InputDims, l3KernelDims, 32, 32, 0.01, rect, ur));
	layers.push_back(new MaxPool(2, l4InputDims, 32, l4PoolDims));
	layers.push_back(new FullyConnected(32 * l5InputDims[0] * l5InputDims[1], 64, 0.01, rect, ur));
	layers.push_back(new FullyConnected(64, numLabels, 0.1, sm, ur));

	FeedForward *ff = new FeedForward(layers, loss);

	cout << "Starting..." << endl;

	nnet_shuffle_instances(features, labels, numTraining + numValidation, numFeatures, numLabels);

	for(size_t i = 0; i < 50; i++)
	{
		nnet_shuffle_instances(features, labels, numTraining, numFeatures, numLabels);

		ff->train(features, labels, numTraining, 1, 100);

		cout << "Epoch: " << (i + 1) << "\t";
		cout << "Training accuracy: " << evaluate(ff, features, labels, numTraining, numFeatures, numLabels) << "\t";
		cout << "Validation Accuracy: " << evaluate(ff, features + numFeatures * numTraining, labels + numLabels * numTraining, numValidation, numFeatures, numLabels) << endl;
	}

	ofstream ofs(argv[2]);
	ff->save(ofs);
	ofs.close();

	for(size_t i = 0; i < layers.size(); i++)
	{
		delete layers[i];
	}

	nnet_free(features);
	nnet_free(labels);
}

