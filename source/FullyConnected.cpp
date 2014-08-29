#include <cstring>
#include <sstream>

#include "nnet/ActivationFunction.hpp"
#include "nnet/core.hpp"
#include "nnet/FullyConnected.hpp"
#include "nnet/UpdateRule.hpp"
#include "nnet/vector.hpp"

using namespace std;

FullyConnected::FullyConnected(size_t inputs, size_t outputs, nnet_float initweight, ActivationFunction *func, UpdateRule *ur)
{
	numInputs = inputs;
	numOutputs = outputs;
	numWeights = inputs * outputs;
	numBiases = outputs;
	initWeight = initweight;
	activationFunction = func;
	updateRule = ur;

	weights = nnet_malloc(numWeights);
	deltaWeights = nnet_malloc(numWeights);
	biases = nnet_malloc(numBiases);
	deltaBiases = nnet_malloc(numBiases);
	activations = nnet_malloc(numOutputs);
	deltaActivations = nnet_malloc(numOutputs);
	deltaErrors = nnet_malloc(numOutputs);
	weightsMomentum = nnet_malloc(numWeights);
	biasesMomentum = nnet_malloc(numBiases);

	//Initialise the weights and biases
	random_gaussian_vector(weights, numWeights, 0.0, initWeight);
	random_gaussian_vector(biases, numBiases, 1.0, initWeight);

	//Set the delta weights and biases to 0
	memset(deltaWeights, 0, sizeof(nnet_float) * numWeights);
	memset(deltaBiases, 0, sizeof(nnet_float) * numBiases);
	memset(weightsMomentum, 0, sizeof(nnet_float) * numWeights);
	memset(biasesMomentum, 0, sizeof(nnet_float) * numBiases);
}

FullyConnected::~FullyConnected()
{
	nnet_free(weights);
	nnet_free(deltaWeights);
	nnet_free(biases);
	nnet_free(deltaBiases);
	nnet_free(activations);
	nnet_free(deltaActivations);
	nnet_free(deltaErrors);
	nnet_free(weightsMomentum);
	nnet_free(biasesMomentum);
}

void FullyConnected::forward(const nnet_float *features)
{
	//Multiply fetures by the weight matrix
	matrix_vector_mul(weights, numOutputs, numInputs, features, activations);

	//Add the biases
	vector_accum(activations, biases, numOutputs);

	//Compute the activation function
	(*activationFunction)(activations, deltaActivations, numOutputs);
}

void FullyConnected::backward(nnet_float *bpDeltaErrors)
{
	//Backpropagate the delta errors to the previous layer
	matrix_trans_vector_mul(weights, numOutputs, numInputs, deltaErrors, bpDeltaErrors);
}

void FullyConnected::calculateGradients(const nnet_float *features)
{
	//Finish calculating the delta errors for this layer
	vector_mul(deltaErrors, deltaActivations, deltaErrors, numOutputs);

	//Now calculate the delta error w.r.t. to each weight
	for(size_t u = 0; u < numOutputs; u++)
	{
		vector_scale_accum(deltaWeights + u * numInputs, features, deltaErrors[u], numInputs);
	}

	//And the delta error w.r.t. each bias
	vector_accum(deltaBiases, deltaErrors, numOutputs);
}

void FullyConnected::updateWeights(const unsigned int batchSize)
{
	updateRule->updateWeights(this, batchSize);
}

void FullyConnected::updateBiases(const unsigned int batchSize)
{
	updateRule->updateBiases(this, batchSize);
}

string FullyConnected::toString() const
{
	stringstream output;
	
	output << "FullyConnected\n"
		<< "\tInputs: " << numInputs << "\n"
		<< "\tOutputs: " << numOutputs << "\n";

	return output.str();
}
