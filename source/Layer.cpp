#include <nnet/Layer.hpp>

using namespace std;

Layer::~Layer()
{

}

void Layer::updateWeights()
{

}

void Layer::updateBiases()
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
