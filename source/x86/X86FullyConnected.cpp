#include "nnet/x86/X86ActivationFunction.hpp"
#include "nnet/x86/X86FullyConnected.hpp"
#include "nnet/x86/X86UpdateRule.hpp"
#include "nnet/x86/vector.hpp"

using namespace std;

X86FullyConnected::X86FullyConnected(size_t inputs, size_t outputs, nnet_float initweight, X86ActivationFunction *func, X86UpdateRule *ur)
{
	numInputs = inputs;
	numOutputs = outputs;
	numWeights = inputs * outputs;
	numBiases = outputs;
	initWeight = initweight;
	activationFunction = func;
	updateRule = ur;
}

void X86FullyConnected::initialise()
{
	//Initialise the weights
	random_gaussian_vector(weights, numWeights, 0.0, initWeight);

	//Initialise the biases
	random_gaussian_vector(biases, numBiases, 0.0, initWeight);
}

void X86FullyConnected::forward(const nnet_float *features)
{
	//Multiply fetures by the weight matrix
	matrix_vector_mul(weights, numOutputs, numInputs, features, activations);

	//Add the biases
	vector_accum(activations, biases, numOutputs);

	//Compute the activation function
	(*activationFunction)(activations, deltaActivations, numOutputs);
}

void X86FullyConnected::backward(nnet_float *bpDeltaErrors)
{
	//Backpropagate the delta errors to the previous layer
	matrix_trans_vector_mul(weights, numOutputs, numInputs, deltaErrors, bpDeltaErrors);
}

void X86FullyConnected::calculateGradients(const nnet_float *features)
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

void X86FullyConnected::updateWeights(const unsigned int batchSize)
{
	updateRule->updateWeights(this, batchSize);
}

void X86FullyConnected::updateBiases(const unsigned int batchSize)
{
	updateRule->updateBiases(this, batchSize);
}
