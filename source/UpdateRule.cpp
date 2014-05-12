#include <nnet/UpdateRule.hpp>

using namespace std;

void SGD::updateWeights(Layer *layer)
{
	for(size_t w = 0; w < layer->weightsSize(); w++)
	{
		layer->weights[w] -= (layer->weights[w] * l2DecayRate + layer->deltaWeights[w]) * learningRate;
		layer->deltaWeights[w] *= momentumRate;
	}
}

void SGD::updateBiases(Layer *layer)
{
	for(size_t b = 0; b < layer->biasesSize(); b++)
	{
		layer->biases[b] -= layer->deltaBiases[b] * learningRate;
		layer->deltaBiases[b] *= momentumRate;
	}
}
