#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <sys/time.h>

#include "nnet/nnet.hpp"
#include "nnet/cifar10.hpp"
#include "evaluate.hpp"

using namespace std;

int main(int argc, char **argv)
{
	ActivationFunction *rect = new RectifiedLinear();
	ActivationFunction *softmax = new Softmax();
	
	SGD *ur = new SGD();
	ur->learningRate = 0.001;
	ur->momentumRate = 0.9;
	
	size_t l1InputDims[] = {32, 32};
	size_t l2InputDims[] = {28, 28};
	size_t l3InputDims[] = {14, 14};
	size_t l4InputDims[] = {10, 10};
	size_t l5InputDims[] = {5, 5};
	size_t kernelDims[] = {5, 5};
	size_t poolDims[] = {2, 2};

	vector<Layer *> layers;
	layers.push_back(new Convolutional(2, l1InputDims, kernelDims, 3, 64, 0.07, rect, ur));
	layers.push_back(new MaxPool(2, l2InputDims, 64, poolDims));
	layers.push_back(new Convolutional(2, l3InputDims, kernelDims, 64, 64, 0.035, rect, ur));
	layers.push_back(new MaxPool(2, l4InputDims, 64, poolDims));
	layers.push_back(new FullyConnected(64 * l5InputDims[0] * l5InputDims[1], 64, 0.035, rect, ur));
	layers.push_back(new FullyConnected(64, 64, 0.1, rect, ur));
	layers.push_back(new FullyConnected(64, 10, 0.15, softmax, ur));

	FeedForward *ff = new FeedForward(layers, new CrossEntropy());

	cout << ff->toString();

	vector<string> filenames;
	vector<nnet_float> features;
	vector<nnet_float> labels;
	
	for(size_t i = 1; i <= 6; i++)
	{
		filenames.push_back(string(argv[i]));
	}

	loadCifar10(filenames, features, labels);

	nnet_float *trainingFeatures = &features[0];
	nnet_float *trainingLabels = &labels[0];
	nnet_float *testingFeatures = &features[32 * 32 * 3 * 50000];
	nnet_float *testingLabels = &labels[10 * 50000];

	size_t numTrainingInsts = 50000;

	for(size_t i = 0; i < 60; i++)
	{
		if(i == 20)
		{
			ur->learningRate = 0.0001;
		}

		nnet_shuffle_instances(trainingFeatures, trainingLabels, numTrainingInsts, 32 * 32 * 3, 10);

		struct timeval start, end;
		gettimeofday(&start, 0);
		ff->train(trainingFeatures, trainingLabels, numTrainingInsts, 1, 100);
		gettimeofday(&end, 0);

		nnet_float duration = (nnet_float)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0);
		nnet_float training = evaluate(ff, trainingFeatures, trainingLabels, numTrainingInsts, 32 * 32 * 3, 10);
		nnet_float test = evaluate(ff, testingFeatures, testingLabels, 10000, 32 * 32 * 3, 10);

		cout << "Epoch: " << i + 1 << "\t";
		cout << "Training duration: " << duration << "\t";
		cout << "Training Accuracy: " << training << "\t";
		cout <<  "Test Accuracy: " << test << endl;
	}

	//ofstream ofs(argv[5]);
	//ff->save(ofs);
	//ofs.close();

	return 0;
}
