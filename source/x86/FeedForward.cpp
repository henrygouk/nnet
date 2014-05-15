#include <algorithm>
#include <cstring>

#include <nnet/core.hpp>
#include <nnet/FeedForward.hpp>
#include <nnet/loss.hpp>

using namespace std;

FeedForward::FeedForward(const vector<Layer *> &layervec, LossFunction lf)
{
	lossFunction = lf;

	//Copy the vector of layers
	layers = layervec;

	numWeights = 0;
	numBiases = 0;
	numActivations = 0;
	numFeatures = layers.front()->inputsSize();
	numLabels = layers.back()->outputsSize();

	//Find out how much memory we need to allocate
	for(size_t l = 0; l < layers.size(); l++)
	{
		numWeights += layers[l]->weightsSize();
		numBiases += layers[l]->biasesSize();
		numActivations += layers[l]->outputsSize();
	}

	//Allocate memory
	nnet_float *weightsPtr = weights = nnet_malloc(numWeights);
	nnet_float *deltaWeightsPtr = deltaWeights = nnet_malloc(numWeights);
	nnet_float *biasesPtr = biases = nnet_malloc(numBiases);
	nnet_float *deltaBiasesPtr = deltaBiases = nnet_malloc(numBiases);
	nnet_float *activationsPtr = activations = nnet_malloc(numActivations);
	nnet_float *deltaActivationsPtr = deltaActivations = nnet_malloc(numActivations);
	nnet_float *deltaErrorsPtr = deltaErrors = nnet_malloc(numActivations);
	memset(deltaWeights, 0, sizeof(nnet_float) * numWeights);
	memset(deltaBiases, 0, sizeof(nnet_float) * numBiases);

	//Set up the pointers
	for(size_t l = 0; l < layers.size(); l++)
	{
		layers[l]->weights = weightsPtr;
		weightsPtr += layers[l]->weightsSize();

		layers[l]->deltaWeights = deltaWeightsPtr;
		deltaWeightsPtr += layers[l]->weightsSize();

		layers[l]->biases = biasesPtr;
		biasesPtr += layers[l]->biasesSize();

		layers[l]->deltaBiases = deltaBiasesPtr;
		deltaBiasesPtr += layers[l]->biasesSize();

		layers[l]->activations = activationsPtr;
		activationsPtr += layers[l]->outputsSize();

		layers[l]->deltaActivations = deltaActivationsPtr;
		deltaActivationsPtr += layers[l]->outputsSize();

		layers[l]->deltaErrors = deltaErrorsPtr;
		deltaErrorsPtr += layers[l]->outputsSize();

		//Initialise this layer
		layers[l]->initialise();
	}

	hypothesis = layers.back()->activations;
}

FeedForward::~FeedForward()
{
	nnet_free(weights);
	nnet_free(deltaWeights);
	nnet_free(biases);
	nnet_free(deltaBiases);
	nnet_free(activations);
	nnet_free(deltaActivations);
	nnet_free(deltaErrors);
}

void FeedForward::train(const nnet_float *features, const nnet_float *labels, const size_t numInstances, uint32_t epochs, uint32_t batchSize)
{
	nnet_float *dErrors = layers.back()->deltaErrors;

	for(uint32_t e = 0; e < epochs; e++)
	{
		for(uint32_t i = 0; i < numInstances; i += batchSize)
		{
			for_each(layers.begin(), layers.end(), [] (Layer *l) { l->startBatch(); });

			for(uint32_t j = 0; j < batchSize; j++)
			{
				forward(features + (i + j) * numFeatures);
				lossFunction(hypothesis, labels + (i + j) * numLabels, dErrors, numLabels);
				backward(features + (i + j) * numFeatures, deltaErrors);
			}

			for_each(layers.begin(), layers.end(), [] (Layer *l) { l->endBatch(); });

			update(batchSize);
		}
	}
}

void FeedForward::predict(const nnet_float *features, nnet_float *labels)
{
	forward(features);

	for(size_t i = 0; i < numLabels; i++)
	{
		labels[i] = hypothesis[i];
	}
}

void FeedForward::forward(const nnet_float *features)
{
	for(size_t l = 0; l < layers.size(); l++)
	{
		layers[l]->forward(features);
		features = layers[l]->activations;
	}
}

void FeedForward::backward(const nnet_float *features, const nnet_float *deltaErrors)
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
	}

	for(size_t l = 0; l < layers.size(); l++)
	{
		layers[l]->updateBiases(batchSize);
	}
}
