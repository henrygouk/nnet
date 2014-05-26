#include "nnet/x86/X86Loss.hpp"

using namespace std;

void X86SquaredError::loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaErrors, size_t numLabels)
{
	for(size_t i = 0; i < numLabels; i++)
	{
		deltaErrors[i] = hypothesis[i] - labels[i];
	}
}

void X86CrossEntropy::loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaErrors, size_t numLabels)
{
	for(size_t i = 0; i < numLabels; i++)
	{
		deltaErrors[i] = (hypothesis[i] - labels[i]);// / (hypothesis[i] * (1.0 - hypothesis[i]));
	}
}
