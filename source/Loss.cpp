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

nnet_float SquaredError::evaluate(const nnet_float *hypotheses, const nnet_float *labels, size_t length, size_t count)
{
	nnet_float total = 0.0;

	for(size_t i = 0; i < count * length; i++)
	{
		nnet_float err = hypotheses[i] - labels[i];
		total += err * err;
	}

	return total;
}

string SquaredError::toString() const
{
	return "SquaredError";
}

void NegativeCosine::loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaErrors, size_t numLabels)
{
	for(size_t i = 0; i < numLabels; i++)
	{
		deltaErrors[i] = -labels[i];
	}
}

nnet_float NegativeCosine::evaluate(const nnet_float *hypotheses, const nnet_float *labels, size_t length, size_t count)
{
	return 0.0;
}

void CrossEntropy::loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaErrors, size_t numLabels)
{
	for(size_t i = 0; i < numLabels; i++)
	{
		deltaErrors[i] = (hypothesis[i] - labels[i]);// / (hypothesis[i] * (1.0 - hypothesis[i]));
	}
}

nnet_float CrossEntropy::evaluate(const nnet_float *hypotheses, const nnet_float *labels, size_t length, size_t count)
{
	return 0.0;
}

string CrossEntropy::toString() const
{
	return "CrossEntropy";
}
