#ifndef _FULLYCONNECTED_HPP_
#define _FULLYCONNECTED_HPP_

#include "X86ActivationFunction.hpp"
#include "X86Layer.hpp"
#include "X86UpdateRule.hpp"

class X86FullyConnected : public X86Layer
{
	public:
		X86FullyConnected(std::size_t inputs, std::size_t outputs, nnet_float initweight, X86ActivationFunction *func, X86UpdateRule *ur);
		void initialise() override;
		void forward(const nnet_float *features) override;
		void backward(nnet_float *bpDeltaErrors) override;
		void calculateGradients(const nnet_float *features) override;
		void updateWeights(const unsigned int batchSize) override;
		void updateBiases(const unsigned int batchSize) override;

	protected:
		nnet_float initWeight;
		X86ActivationFunction *activationFunction;
		X86UpdateRule *updateRule;
};

#endif
