#include <fstream>

#include "nnet/Layer.hpp"

using namespace std;

Layer::~Layer()
{

}

void Layer::save(ostream &os)
{
	os.write((char *)weights, sizeof(nnet_float) * numWeights);
	os.write((char *)biases, sizeof(nnet_float) * numBiases);
}

void Layer::load(istream &is)
{
	is.read((char *)weights, sizeof(nnet_float) * numWeights);
	is.read((char *)biases, sizeof(nnet_float) * numBiases);
}

void Layer::forwardTrain(const nnet_float *features)
{
	forward(features);
}

void Layer::calculateGradients(const nnet_float *features)
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

string Layer::toString() const
{
	return "Layer\n";
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
