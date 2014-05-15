#include <nnet/Layer.hpp>

using namespace std;

Layer::~Layer()
{

}

void Layer::updateWeights(const unsigned int batchSize)
{

}

void Layer::updateBiases(const unsigned int batchSize)
{

}

void Layer::startBatch()
{

}

void Layer::endBatch()
{

}

size_t Layer::weightsSize() const
{
	return numWeights;
}

size_t Layer::biasesSize() const
{
	return numBiases;
}

size_t Layer::inputsSize() const
{
	return numInputs;
}

size_t Layer::outputsSize() const
{
	return numOutputs;
}
