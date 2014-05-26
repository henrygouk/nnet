#include <cstdlib>

#include "nnet/x86/X86Dropout.hpp"

X86Dropout::X86Dropout(size_t numinputs, nnet_float prob)
{
	dropoutProbability = prob;
	forwardScalar = (1.0 - prob);
	numInputs = numinputs;
	numOutputs = numinputs;
	numWeights = 0;
	numBiases = 0;
}

void X86Dropout::initialise()
{
	
}

void X86Dropout::forwardTrain(const nnet_float *features)
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

void X86Dropout::forward(const nnet_float *features)
{
	for(size_t i = 0; i < numInputs; i++)
	{
		activations[i] = forwardScalar * features[i];
	}
}

void X86Dropout::backward(nnet_float *bpDeltaErrors)
{
	for(size_t i = 0; i < numInputs; i++)
	{
		bpDeltaErrors[i] = deltaErrors[i];
	}
}
