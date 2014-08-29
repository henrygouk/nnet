#include <cstdlib>
#include <sstream>

#include "nnet/core.hpp"
#include "nnet/Dropout.hpp"

using namespace std;

Dropout::Dropout(size_t numinputs, nnet_float prob)
{
	dropoutProbability = prob;
	forwardScalar = (1.0 - prob);
	numInputs = numinputs;
	numOutputs = numinputs;
	numWeights = 0;
	numBiases = 0;

	activations = nnet_malloc(numOutputs);
	deltaActivations = nnet_malloc(numOutputs);
	deltaErrors = nnet_malloc(numOutputs);
}

Dropout::~Dropout()
{
	nnet_free(activations);
	nnet_free(deltaActivations);
	nnet_free(deltaErrors);
}

void Dropout::forwardTrain(const nnet_float *features)
{
	for(size_t i = 0; i < numInputs; i++)
	{
		if(((float)rand() / (float)RAND_MAX) <= dropoutProbability)
		{
			activations[i] = 0.0;
		}
		else
		{
			activations[i] = features[i];
		}
	}
}

void Dropout::forward(const nnet_float *features)
{
	for(size_t i = 0; i < numInputs; i++)
	{
		activations[i] = forwardScalar * features[i];
	}
}

void Dropout::backward(nnet_float *bpDeltaErrors)
{
	for(size_t i = 0; i < numInputs; i++)
	{
		bpDeltaErrors[i] = deltaErrors[i];
	}
}

string Dropout::toString() const
{
	stringstream output;

	output << "Dropout\n"
		<< "\tPropability: " << dropoutProbability << "\n"
		<< "\tUnits: " << numInputs << "\n";

	return output.str();
}
