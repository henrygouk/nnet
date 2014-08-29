#include <cmath>
#include <cfloat>

#include "nnet/UpdateRule.hpp"

using namespace std;

SGD::SGD()
{
	maxNorm = FLT_MAX;
}

void SGD::updateWeights(Layer *layer, const unsigned int batchSize)
{
	nnet_float scalar = 1.0 / (nnet_float)batchSize;
	nnet_float norm = 0.0;

	for(size_t w = 0; w < layer->weightsSize(); w++)
	{
		layer->weightsMomentum[w] = layer->weightsMomentum[w] * momentumRate + (layer->weights[w] * l2DecayRate + layer->deltaWeights[w] * scalar) * learningRate;
		layer->weights[w] -= layer->weightsMomentum[w];
		layer->deltaWeights[w] = 0.0;
		
		norm += layer->weights[w] * layer->weights[w];
	}

	norm = sqrt(norm);

	if(norm > maxNorm)
	{
		norm = maxNorm / norm;

		for(size_t w = 0; w < layer->weightsSize(); w++)
		{
			layer->weights[w] *= norm;
		}
	}
}

void SGD::updateBiases(Layer *layer, const unsigned int batchSize)
{
	nnet_float scalar = learningRate / (nnet_float)batchSize;

	for(size_t b = 0; b < layer->biasesSize(); b++)
	{
		layer->biasesMomentum[b] = layer->biasesMomentum[b] * momentumRate + layer->deltaBiases[b] * scalar;
		layer->biases[b] -= layer->biasesMomentum[b];
		layer->deltaBiases[b] = 0.0;
	}
}
