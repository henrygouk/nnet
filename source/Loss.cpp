#include "nnet/Loss.hpp"

using namespace std;

string Loss::toString() const
{
	return "Loss";
}

void SquaredError::loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaErrors, size_t numLabels)
{
	for(size_t i = 0; i < numLabels; i++)
	{
		deltaErrors[i] = hypothesis[i] - labels[i];
	}
}

string SquaredError::toString() const
{
	return "SquaredError";
}

void CrossEntropy::loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaErrors, size_t numLabels)
{
	for(size_t i = 0; i < numLabels; i++)
	{
		deltaErrors[i] = (hypothesis[i] - labels[i]);// / (hypothesis[i] * (1.0 - hypothesis[i]));
	}
}

string CrossEntropy::toString() const
{
	return "CrossEntropy";
}
