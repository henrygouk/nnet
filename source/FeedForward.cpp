#include <algorithm>
#include <cstring>
#include <iostream>

#include "nnet/core.hpp"
#include "nnet/FeedForward.hpp"
#include "nnet/Loss.hpp"

using namespace std;

FeedForward::FeedForward(const vector<Layer *> &layervec, Loss *lf)
{
	lossFunction = lf;

	//Copy the vector of layers
	layers = layervec;

	numFeatures = layers.front()->inputsSize();
	numLabels = layers.back()->outputsSize();
	hypothesis = layers.back()->activations;
}

FeedForward::~FeedForward()
{
	
}

void FeedForward::load(istream &is)
{
	for(size_t i = 0; i < layers.size(); i++)
	{
		layers[i]->load(is);
	}
}

void FeedForward::save(ostream &os)
{
	for(size_t i = 0; i < layers.size(); i++)
	{
		layers[i]->save(os);
	}
}

void FeedForward::train(const nnet_float *features, const nnet_float *labels, const size_t numInstances, uint32_t epochs, uint32_t batchSize)
{
	nnet_float *dErrors = layers.back()->deltaErrors;

	for(uint32_t e = 0; e < epochs; e++)
	{
		for(uint32_t i = 0; i < numInstances; i += batchSize)
		{
			for_each(layers.begin(), layers.end(), [] (Layer *l) { l->startBatch(); });

			for(uint32_t j = 0; j < batchSize && (i + j) < numInstances; j++)
			{
				forward(features + (i + j) * numFeatures);
				lossFunction->loss(hypothesis, labels + (i + j) * numLabels, dErrors, numLabels);
				backward(features + (i + j) * numFeatures);
			}

			for_each(layers.begin(), layers.end(), [] (Layer *l) { l->endBatch(); });

			update(batchSize);
		}
	}
}

void FeedForward::predict(const nnet_float *features, nnet_float *labels)
{
	for(size_t l = 0; l < layers.size(); l++)
	{
		layers[l]->forward(features);
		features = layers[l]->activations;
	}

	for(size_t i = 0; i < numLabels; i++)
	{
		labels[i] = hypothesis[i];
	}
}

void FeedForward::forward(const nnet_float *features)
{
	for(size_t l = 0; l < layers.size(); l++)
	{
		layers[l]->forwardTrain(features);
		features = layers[l]->activations;
	}
}

void FeedForward::backward(const nnet_float *features)
{
	for(size_t l = layers.size() - 1; l > 0; l--)
	{
		layers[l]->calculateGradients(layers[l - 1]->activations);
		layers[l]->backward(layers[l - 1]->deltaErrors);
	}

	layers[0]->calculateGradients(features);
}

void FeedForward::update(const unsigned int batchSize)
{
	for(size_t l = 0; l < layers.size(); l++)
	{
		layers[l]->updateWeights(batchSize);
		layers[l]->updateBiases(batchSize);
	}
}

string FeedForward::toString() const
{
	string output = "FeedForward\n";
	output += "\tLoss: " + lossFunction->toString() + "\n\n";

	for(size_t i = 0; i < layers.size(); i++)
	{
		output += layers[i]->toString() + "\n";
	}

	return output;
}

size_t FeedForward::outputsSize() const
{
	return layers[layers.size() - 1]->outputsSize();
}

const vector<Layer *> &FeedForward::getLayers() const
{
	return layers;
}
