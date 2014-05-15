#include <nnet/core.hpp>
#include <nnet/MaxPool.hpp>
#include <nnet/vector.hpp>
#include <cstring>
#include <cfloat>

using namespace std;

MaxPool::MaxPool(size_t rank, const size_t *inputDims, size_t chans, const size_t *poolDims)
{
	tensorRank = rank;
	inputDimensions = new size_t[rank];
	poolDimensions = new size_t[rank];
	outputDimensions = new size_t[rank];
	channels = chans;

	memcpy(inputDimensions, inputDims, sizeof(size_t) * rank);
	memcpy(poolDimensions, poolDims, sizeof(size_t) * rank);

	inputVolume = 1;
	outputVolume = 1;

	for(size_t i = 0; i < rank; i++)
	{
		outputDimensions[i] = inputDimensions[i] / poolDimensions[i];
		inputVolume *= inputDimensions[i];
		outputVolume *= outputDimensions[i];
	}

	numWeights = 0;
	numBiases = 0;
	numOutputs = outputVolume * channels;
	numInputs = inputVolume * channels;
	inputIndices = new size_t[outputVolume * channels];
}

void MaxPool::initialise()
{

}

void MaxPool::forward(const nnet_float *features)
{
	nnet_float *output = activations;
	
	for(size_t i = 0; i < numOutputs; i++)
	{
		output[i] = -FLT_MAX;
	}

	for(size_t c = 0; c < channels; c++)
	{
		tensor_maxpool_nd(tensorRank, features, inputDimensions, poolDimensions, output, inputIndices + c * outputVolume, c * inputVolume);
		features += inputVolume;
		output += outputVolume;
	}
}

void MaxPool::backward(nnet_float *bpDeltaErrors)
{
	memset(bpDeltaErrors, 0, sizeof(nnet_float) * numInputs);

	for(size_t i = 0; i < numOutputs; i++)
	{
		bpDeltaErrors[inputIndices[i]] = deltaErrors[i];
	}
}

void MaxPool::calculateGradients(const nnet_float *features)
{
	
}
