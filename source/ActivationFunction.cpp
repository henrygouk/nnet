#include <cfloat>
#include <cmath>

#include "nnet/ActivationFunction.hpp"

using namespace std;

void Logistic::operator()(nnet_float *activations, nnet_float *deltaActivations, size_t numActivations) const
{
	for(size_t i = 0; i < numActivations; i++)
	{
		activations[i] = 1.0 / (1.0 + exp(-activations[i]));
		deltaActivations[i] = activations[i] * (1.0 - activations[i]);
	}
}

void Softmax::operator()(nnet_float *activations, nnet_float *deltaActivations, size_t numActivations) const
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

SlidingSoftmax::SlidingSoftmax(size_t numlbls, size_t stride)
{
	this->numLabels = numlbls;
	this->stride = stride;
}

void SlidingSoftmax::operator()(nnet_float *activations, nnet_float *deltaActivations, size_t numActivations) const
{
	for(size_t s = 0; s < stride; s++)
	{
		nnet_float maxval = -FLT_MAX;
		nnet_float sum = 0.0;

		for(size_t i = 0; i < numLabels; i++)
		{
			maxval = max(activations[i * stride + s], maxval);
		}

		for(size_t i = 0; i < numLabels; i++)
		{
			activations[i * stride + s] = exp(activations[i * stride + s] - maxval);
			sum += activations[i * stride + s];
		}

		sum = 1.0 / sum;

		for(size_t i = 0; i < numLabels; i++)
		{
			activations[i * stride + s] *= sum;
		}
	}
}

void RectifiedLinear::operator()(nnet_float *activations, nnet_float *deltaActivations, size_t numActivations) const
{
	for(size_t i = 0; i < numActivations; i++)
	{
		deltaActivations[i] = activations[i] >= 0.0 ? 1.0 : 0.0;
		activations[i] = activations[i] >= 0.0 ? activations[i] : 0.0;
	}
}

