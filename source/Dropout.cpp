#include <cstdlib>
#include <sstream>

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
}

void Dropout::initialise()
{
	
}

void Dropout::forwardTrain(const nnet_float *features)
{
	for(size_t i = 0; i < numInputs; i++)
	{
		if(rand() <= dropoutProbability)
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
		<< "\tPropability: " << dropoutProbability << "\n";

	return output.str();
}
