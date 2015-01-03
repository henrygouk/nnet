#include <iostream>
#include <fstream>
#include <ctime>
#include <sys/time.h>

#include <nnet/nnet.hpp>
#include "mnist.hpp"
#include "evaluate.hpp"

using namespace std;

int main(int argc, char **argv)
{
	size_t numTrainingInsts = 10000;//60000;

	ActivationFunction *rect = new RectifiedLinear();
	ActivationFunction *softmax = new Softmax();
	
	SGD *ur = new SGD();
	ur->learningRate = 0.01;
	ur->momentumRate = 0.9;
	
	size_t l1InputDims[] = {28, 28};
	size_t l2InputDims[] = {20, 20};
	size_t l3InputDims[] = {4, 4};
	size_t kernelDims[] = {9, 9};
	size_t poolDims[] = {5, 5};
	size_t chans = 112;

	vector<Layer *> layers;
	layers.push_back(new Convolutional(2, l1InputDims, kernelDims, 1, chans, 0.01, rect, ur));
	//layers.push_back(new SpatialConvolutional(l1InputDims, kernelDims, 1, chans, 0.01, rect, ur));
	layers.push_back(new MaxPool(2, l2InputDims, chans, poolDims));
	layers.push_back(new FullyConnected(chans * l3InputDims[0] * l3InputDims[1], 10, 0.01, softmax, ur));

	FeedForward *ff = new FeedForward(layers, new CrossEntropy());

	cout << ff->toString();

	nnet_float *trainingFeatures = mnist_training_images(argv[1]);
	nnet_float *trainingLabels = mnist_training_labels(argv[2]);
	nnet_float *testingFeatures = mnist_testing_images(argv[3]);
	nnet_float *testingLabels = mnist_testing_labels(argv[4]);

	for(size_t i = 0; i < 60; i++)
	{
		if(i == 20)
		{
			ur->learningRate = 0.001;
		}

		nnet_shuffle_instances(trainingFeatures, trainingLabels, numTrainingInsts, 28 * 28, 10);

		struct timeval start, end;
		gettimeofday(&start, 0);
		ff->train(trainingFeatures, trainingLabels, numTrainingInsts, 1, 100);
		gettimeofday(&end, 0);

		nnet_float duration = (nnet_float)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0);
		//nnet_float validation = evaluate(ff, trainingFeatures + numTrainingInsts * 28 * 28, trainingLabels + numTrainingInsts * 10, 10000, 28 * 28, 10);
		nnet_float training = 0;//evaluate(ff, trainingFeatures, trainingLabels, numTrainingInsts, 28 * 28, 10);
		nnet_float test = 0;//evaluate(ff, testingFeatures, testingLabels, 10000, 28 * 28, 10);

		cout << "Epoch: " << i + 1 << "\t";
		cout << "Training duration: " << duration << "\t";
		cout << "Training Accuracy: " << training << "\t";
		//cout << "Validation: " << validation << endl;
		cout <<  "Test Accuracy: " << test << endl;
	}

	ofstream ofs(argv[5]);
	ff->save(ofs);
	ofs.close();

	nnet_free(trainingFeatures);
	nnet_free(trainingLabels);
	nnet_free(testingFeatures);
	nnet_free(testingLabels);

	return 0;
}
