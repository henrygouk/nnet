#include <cstring>

#include "nnet/SpatialConvolutional.hpp"
#include "nnet/vector.hpp"
#include "nnet/core.hpp"

using namespace std;

SpatialConvolutional::SpatialConvolutional(const size_t *inputDims, const size_t *kernelDims, size_t inputs, size_t outputs, nnet_float initweight, ActivationFunction *func, UpdateRule *ur)
{
	this->inputDims = new size_t[2];
	this->kernelDims = new size_t[2];
	this->outputDims = new size_t[2];

	for(size_t i = 0; i < 2; i++)
	{
		this->inputDims[i] = inputDims[i];
		this->kernelDims[i] = kernelDims[i];
		this->outputDims[i] = (inputDims[i] - kernelDims[i] + 1);
	}

	numInputChannels = inputs;
	numOutputChannels = outputs;
	inputVolume = inputDims[0] * inputDims[1];
	kernelVolume = kernelDims[0] * kernelDims[1];
	outputVolume = outputDims[0] * outputDims[1];
	numInputs = inputs * inputVolume;
	numOutputs = outputs * outputVolume;
	numWeights = inputs * outputs * kernelVolume;
	numBiases = outputs;
	initWeight = initweight;
	updateRule = ur;
	activationFunction = func;

	weights = nnet_malloc(numWeights);
	deltaWeights = nnet_malloc(numWeights);
	biases = nnet_malloc(numBiases);
	deltaBiases = nnet_malloc(numBiases);
	activations = nnet_malloc(numOutputs);
	deltaActivations = nnet_malloc(numOutputs);
	deltaErrors = nnet_malloc(numOutputs);
	weightsMomentum = nnet_malloc(numWeights);
	biasesMomentum = nnet_malloc(numBiases);

	//Initialise the weights
	random_gaussian_vector(weights, numWeights, 0.0, initWeight);

	//Initialise the biases
	random_gaussian_vector(biases, numBiases, 1.0, initWeight);

	memset(deltaWeights, 0, sizeof(nnet_float) * numWeights);
	memset(deltaBiases, 0, sizeof(nnet_float) * numBiases);
	memset(weightsMomentum, 0, sizeof(nnet_float) * numWeights);
	memset(biasesMomentum, 0, sizeof(nnet_float) * numBiases);
}

SpatialConvolutional::~SpatialConvolutional()
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

	delete[] inputDims;
	delete[] kernelDims;
	delete[] outputDims;
}

void SpatialConvolutional::endBatch()
{
	for(size_t i = 0; i < numInputChannels * numOutputChannels; i++)
	{
		reverse(deltaWeights + i * kernelVolume, kernelVolume);
	}
}

void SpatialConvolutional::forward(const nnet_float *features)
{
	nnet_float *as = activations;
	nnet_float *ws = weights;

	for(size_t i = 0; i < numOutputChannels; i++)
	{
		const nnet_float *fs = features;

		for(size_t j = 0; j < outputVolume; j++)
		{
			as[j] = biases[i];
		}

		for(size_t j = 0; j < numInputChannels; j++)
		{
			convolve_valid(fs, inputDims, ws, kernelDims, as);
			ws += kernelVolume;
			fs += inputVolume;
		}

		as += outputVolume;
	}

	(*activationFunction)(activations, deltaActivations, numOutputs);
}

void SpatialConvolutional::backward(nnet_float *bpDeltaErrors)
{
	memset(bpDeltaErrors, 0, sizeof(nnet_float) * numInputChannels * inputVolume);

	for(size_t i = 0; i < numInputChannels; i++)
	{
		nnet_float *errs = deltaErrors;
		nnet_float *ws = weights + i * kernelVolume;

		for(size_t j = 0; j < numOutputChannels; j++)
		{
			correlate_full(errs, outputDims, ws, kernelDims, bpDeltaErrors);

			ws += kernelVolume * numInputChannels;
			errs += outputVolume;
		}

		bpDeltaErrors += inputVolume;
	}
}

void SpatialConvolutional::calculateGradients(const nnet_float *features)
{
	nnet_float *des = deltaErrors;
	nnet_float *dws = deltaWeights;

	vector_mul(deltaErrors, deltaActivations, deltaErrors, numOutputs);

	for(size_t i = 0; i < numOutputChannels; i++)
	{
		const nnet_float *fs = features;

		deltaBiases[i] += vector_sum(des, outputVolume);

		for(size_t j = 0; j < numInputChannels; j++)
		{
			correlate_valid(fs, inputDims, des, outputDims, dws);
			dws += kernelVolume;
			fs += inputVolume;
		}

		des += outputVolume;
	}
}

void SpatialConvolutional::updateWeights(const unsigned int batchSize)
{
	updateRule->updateWeights(this, batchSize);
}

void SpatialConvolutional::updateBiases(const unsigned int batchSize)
{
	updateRule->updateBiases(this, batchSize);
}
