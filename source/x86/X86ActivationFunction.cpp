#include <cfloat>
#include <cmath>

#include "nnet/x86/X86ActivationFunction.hpp"

using namespace std;

void X86Logistic::operator()(nnet_float *activations, nnet_float *deltaActivations, size_t numActivations) const
{
	for(size_t i = 0; i < numActivations; i++)
	{
		activations[i] = 1.0 / (1.0 + exp(-activations[i]));
		deltaActivations[i] = activations[i] * (1.0 - activations[i]);
	}
}

void X86Softmax::operator()(nnet_float *activations, nnet_float *deltaActivations, size_t numActivations) const
{
	nnet_float maxval = -FLT_MAX;
	nnet_float sum = 0.0;

	for(size_t i = 0; i < numActivations; i++)
	{
		maxval = max(activations[i], maxval);
	}

	for(size_t i = 0; i < numActivations; i++)
	{
		activations[i] = exp(activations[i] - maxval);
		sum += activations[i];
	}

	sum = 1.0 / sum;

	for(size_t i = 0; i < numActivations; i++)
	{
		activations[i] *= sum;
		deltaActivations[i] = 1.0;//activations[i] * (1.0 - activations[i]);
	}
}

void X86RectifiedLinear::operator()(nnet_float *activations, nnet_float *deltaActivations, size_t numActivations) const
{
	for(size_t i = 0; i < numActivations; i++)
	{
		deltaActivations[i] = activations[i] >= 0.0 ? 1.0 : 0.0;
		activations[i] = activations[i] >= 0.0 ? activations[i] : 0.0;
	}
}

