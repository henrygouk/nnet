#ifndef _FULLYCONNECTED_HPP_
#define _FULLYCONNECTED_HPP_

#include "Layer.hpp"

class ActivationFunction;
class UpdateRule;

class FullyConnected : public Layer
{
	public:
		FullyConnected(std::size_t inputs, std::size_t outputs, nnet_float initweight, const ActivationFunction *func, UpdateRule *ur);
		void initialise() override;
		void forward(const nnet_float *features) override;
		void backward(nnet_float *bpDeltaErrors) override;
		void calculateGradients(const nnet_float *features) override;
		void updateWeights(const unsigned int batchSize) override;
		void updateBiases(const unsigned int batchSize) override;

	protected:
		nnet_float initWeight;
		const ActivationFunction *activationFunction;
		UpdateRule *updateRule;
};

#endif
