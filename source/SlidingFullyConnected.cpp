#include <cstring>
#include <sstream>

#include "nnet/ActivationFunction.hpp"
#include "nnet/core.hpp"
#include "nnet/SlidingFullyConnected.hpp"
#include "nnet/UpdateRule.hpp"
#include "nnet/vector.hpp"

using namespace std;

SlidingFullyConnected::SlidingFullyConnected(size_t rank, size_t *inputDims, size_t inputChannels, size_t outputChannels, nnet_float initweight, ActivationFunction *func, UpdateRule *ur)
{
	volume = 1;
	numInputChannels = inputChannels;
	numOutputChannels = outputChannels;
	numInputs = inputChannels;
	numOutputs = outputChannels;
	numWeights = inputChannels * outputChannels;
	numBiases = outputChannels;
	initWeight = initweight;
	activationFunction = func;
	updateRule = ur;

	for(size_t i = 0; i < rank; i++)
	{
		volume *= inputDims[i];
	}

	numInputs *= volume;
	numOutputs *= volume;

	weights = nnet_malloc(numWeights);
	deltaWeights = nnet_malloc(numWeights);
	biases = nnet_malloc(numBiases);
	deltaBiases = nnet_malloc(numBiases);
	activations = nnet_malloc(numOutputs);
	deltaActivations = nnet_malloc(numOutputs);
	deltaErrors = nnet_malloc(numOutputs);
	weightsMomentum = nnet_malloc(numWeights);
	biasesMomentum = nnet_malloc(numBiases);

	//Initialise the weights and biases
	random_gaussian_vector(weights, numWeights, 0.0, initWeight);
	random_gaussian_vector(biases, numBiases, 1.0, initWeight);

	//Set the delta weights and biases to 0
	memset(deltaWeights, 0, sizeof(nnet_float) * numWeights);
	memset(deltaBiases, 0, sizeof(nnet_float) * numBiases);
	memset(weightsMomentum, 0, sizeof(nnet_float) * numWeights);
	memset(biasesMomentum, 0, sizeof(nnet_float) * numBiases);
}

SlidingFullyConnected::~SlidingFullyConnected()
{
	nnet_free(weights);
	nnet_free(deltaWeights);
	nnet_free(biases);
	nnet_free(deltaBiases);
	nnet_free(activations);
	nnet_free(deltaActivations);
	nnet_free(deltaErrors);
	nnet_free(weightsMomentum);
	nnet_free(biasesMomentum);
}

void SlidingFullyConnected::forward(const nnet_float *features)
{
	//Multiply fetures by the weight matrix
	//matrix_vector_mul(weights, numOutputs, numInputs, features, activations);

	nnet_float *as = activations;
	nnet_float *ws = weights;

	//Iterate over each output channel
	for(size_t o = 0; o < numOutputChannels; o++)
	{
		const nnet_float *fs = features;
		
		for(size_t i = 0; i < volume; i++)
		{
			as[i] = biases[o];
		}

		//Iterate over each input channel
		for(size_t i = 0; i < numInputChannels; i++)
		{
			vector_scale_accum(as, fs, *ws, volume);

			fs += volume;
			ws++;
		}

		as += volume;
	}

	//Compute the activation function
	(*activationFunction)(activations, deltaActivations, numOutputs);
}

void SlidingFullyConnected::backward(nnet_float *bpDeltaErrors)
{
	for(size_t i = 0; i < numInputChannels; i++)
	{
		memset(bpDeltaErrors, 0, sizeof(nnet_float) * volume);

		for(size_t j = 0; j < numOutputChannels; j++)
		{
			vector_scale_accum(bpDeltaErrors, deltaErrors + j * volume, weights[j * numInputChannels + i], volume);
		}

		bpDeltaErrors += volume;
	}
}

void SlidingFullyConnected::calculateGradients(const nnet_float *features)
{
	vector_mul(deltaErrors, deltaActivations, deltaErrors, numOutputs);

	for(size_t i = 0; i < numOutputChannels; i++)
	{
		deltaBiases[i] += vector_sum(deltaErrors + i * volume, volume);

		for(size_t j = 0; j < numInputChannels; j++)
		{
			deltaWeights[i * numInputChannels + j] += dot_product(features + j * volume, deltaErrors + i * volume, volume);
		}
	}
}

void SlidingFullyConnected::updateWeights(const unsigned int batchSize)
{
	updateRule->updateWeights(this, batchSize);
}

void SlidingFullyConnected::updateBiases(const unsigned int batchSize)
{
	updateRule->updateBiases(this, batchSize);
}

string SlidingFullyConnected::toString() const
{
	stringstream output;
	
	output << "SlidingFullyConnected\n"
		<< "\tInputs: " << numInputs << "\n"
		<< "\tOutputs: " << numOutputs << "\n";

	return output.str();
}
