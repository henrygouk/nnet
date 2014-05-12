#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include "types.hpp"

class Layer
{
	public:
		virtual ~Layer();
		virtual void initialise() = 0;
		virtual void forward(const nnet_float *features) = 0;
		virtual void backward(nnet_float *bpDeltaErrors) = 0;
		virtual void calculateGradients(const nnet_float *features) = 0;
		virtual void updateWeights();
		virtual void updateBiases();
		virtual void startBatch();
		virtual void endBatch();
		virtual std::size_t weightsSize() const;
		virtual std::size_t biasesSize() const;
		virtual std::size_t inputsSize() const;
		virtual std::size_t outputsSize() const;

		nnet_float *weights;
		nnet_float *deltaWeights;
		nnet_float *biases;
		nnet_float *deltaBiases;
		nnet_float *activations;
		nnet_float *deltaActivations;
		nnet_float *deltaErrors;

	protected:
		std::size_t numWeights;
		std::size_t numBiases;
		std::size_t numInputs;
		std::size_t numOutputs;
};

#endif
