#include "nnet/x86/X86Layer.hpp"

using namespace std;

X86Layer::~X86Layer()
{

}

void X86Layer::forwardTrain(const nnet_float *features)
{
	forward(features);
}

void X86Layer::calculateGradients(const nnet_float *features)
{

}

void X86Layer::updateWeights(const unsigned int batchSize)
{

}

void X86Layer::updateBiases(const unsigned int batchSize)
{

}

void X86Layer::startBatch()
{

}

void X86Layer::endBatch()
{

}

size_t X86Layer::weightsSize() const
{
	return numWeights;
}

size_t X86Layer::biasesSize() const
{
	return numBiases;
}

size_t X86Layer::inputsSize() const
{
	return numInputs;
}

size_t X86Layer::outputsSize() const
{
	return numOutputs;
}
