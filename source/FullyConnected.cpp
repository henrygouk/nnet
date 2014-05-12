#include <nnet/ActivationFunction.hpp>
#include <nnet/FullyConnected.hpp>
#include <nnet/UpdateRule.hpp>
#include <nnet/vector.hpp>

using namespace std;

FullyConnected::FullyConnected(size_t inputs, size_t outputs, nnet_float initweight, const ActivationFunction *func, UpdateRule *ur)
{
	numInputs = inputs;
	numOutputs = outputs;
	numWeights = inputs * outputs;
	numBiases = outputs;
	initWeight = initweight;
	activationFunction = func;
	updateRule = ur;
}

void FullyConnected::initialise()
{
	//Initialise the weights
	random_vector(weights, numWeights, -initWeight, initWeight);

	//Initialise the biases
	random_vector(biases, numBiases, -initWeight, initWeight);
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

void FullyConnected::updateWeights()
{
	updateRule->updateWeights(this);
}

void FullyConnected::updateBiases()
{
	updateRule->updateBiases(this);
}
