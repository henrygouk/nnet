#include "nnet/x86/X86UpdateRule.hpp"

using namespace std;

void X86SGD::updateWeights(X86Layer *layer, const unsigned int batchSize)
{
	nnet_float scalar = 1.0 / (nnet_float)batchSize;

	for(size_t w = 0; w < layer->weightsSize(); w++)
	{
		layer->weights[w] -= (layer->weights[w] * l2DecayRate + layer->deltaWeights[w] * scalar) * learningRate;
		layer->deltaWeights[w] *= momentumRate;
	}
}

void X86SGD::updateBiases(X86Layer *layer, const unsigned int batchSize)
{
	nnet_float scalar = learningRate / (nnet_float)batchSize;

	for(size_t b = 0; b < layer->biasesSize(); b++)
	{
		layer->biases[b] -= layer->deltaBiases[b] * scalar;
		layer->deltaBiases[b] *= momentumRate;
	}
}
