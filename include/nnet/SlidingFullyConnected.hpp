#ifndef _SLIDINGFULLYCONNECTED_HPP_
#define _SLIDINGFULLYCONNECTED_HPP_

#include "ActivationFunction.hpp"
#include "Layer.hpp"
#include "UpdateRule.hpp"

class SlidingFullyConnected : public Layer
{
	public:
		SlidingFullyConnected(std::size_t rank, std::size_t *inputDims, std::size_t inputChannels, std::size_t outputChannels, nnet_float initweight, ActivationFunction *func, UpdateRule *ur);
		~SlidingFullyConnected();
		void forward(const nnet_float *features) override;
		void backward(nnet_float *bpDeltaErrors) override;
		void calculateGradients(const nnet_float *features) override;
		void updateWeights(const unsigned int batchSize) override;
		void updateBiases(const unsigned int batchSize) override;
		std::string toString() const override;

	protected:
		std::size_t volume;
		std::size_t numInputChannels;
		std::size_t numOutputChannels;
		nnet_float initWeight;
		ActivationFunction *activationFunction;
		UpdateRule *updateRule;
};

#endif
