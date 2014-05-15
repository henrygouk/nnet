#include "nnet/types.hpp"

using namespace std;

void squaredError(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaErrors, size_t numLabels)
{
	for(size_t i = 0; i < numLabels; i++)
	{
		deltaErrors[i] = hypothesis[i] - labels[i];
	}
}

void crossEntropy(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaErrors, size_t numLabels)
{
	for(size_t i = 0; i < numLabels; i++)
	{
		deltaErrors[i] = (hypothesis[i] - labels[i]) / (hypothesis[i] * (1.0 - hypothesis[i]));
	}
}
