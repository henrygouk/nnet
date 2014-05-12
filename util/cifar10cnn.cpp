#include <iostream>

#include <nnet/nnet.hpp>

#include "cifar10.hpp"
#include "evaluate.hpp"

using namespace std;

int main(int argc, char **argv)
{
	ActivationFunction *rect = new RectifiedLinear();
	ActivationFunction *softmax = new Softmax();
	SGD *ur = new SGD();
	ur->learningRate = 0.001;
	ur->momentumRate = 0.9;
	ur->l2DecayRate = 0.0;

	size_t l1InputDims[] = {32, 32};
	size_t l2InputDims[] = {28, 28};
	size_t l3InputDims[] = {14, 14};
	size_t l4InputDims[] = {10, 10};
	size_t l5InputDims[] = {5, 5};
	size_t kernelDims[] = {5, 5};
	size_t poolDims[] = {2, 2};

	vector<Layer *> layers;
	layers.push_back(new Convolutional(2, l1InputDims, kernelDims, 3, 64, 0.02, rect, ur));
	layers.push_back(new MaxPool(2, l2InputDims, 64, poolDims));
	layers.push_back(new Convolutional(2, l3InputDims, kernelDims, 64, 64, 0.02, rect, ur));
	layers.push_back(new MaxPool(2, l4InputDims, 64, poolDims));
	layers.push_back(new Convolutional(2, l5InputDims, kernelDims, 64, 64, 0.08, rect, ur));
	layers.push_back(new FullyConnected(64, 64, 0.08, rect, ur));
	layers.push_back(new FullyConnected(64, 10, 0.08, softmax, ur));

	FeedForward *ff = new FeedForward(layers, &crossEntropy);

	nnet_float *features, *labels;
	cifar10(argv[1], &features, &labels);

	ff->train(features, labels, 40000, 1, 100);

	cout << evaluate(ff, features + 40000 * 32 * 32 * 3, labels + 40000 * 10, 10000, 32 * 32 * 3, 10) << endl;

	ff->train(features, labels, 40000, 1, 100);

	cout << evaluate(ff, features + 40000 * 32 * 32 * 3, labels + 40000 * 10, 10000, 32 * 32 * 3, 10) << endl;
}
