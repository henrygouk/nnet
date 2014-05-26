#ifndef _CONVOLUTIONAL_HPP_
#define _CONVOLUTIONAL_HPP_

#include "X86ActivationFunction.hpp"
#include "X86Layer.hpp"
#include "X86UpdateRule.hpp"

#include <fftw3.h>

class X86Convolutional : public X86Layer
{
	public:
		X86Convolutional(std::size_t rank, const std::size_t *imageDims, const std::size_t *kernelDims, std::size_t inputs, std::size_t outputs, nnet_float initweight, X86ActivationFunction *func, X86UpdateRule *ur);
		void initialise() override;
		void startBatch() override;
		void endBatch() override;
		void forward(const nnet_float *features) override;
		void backward(nnet_float *bpDeltaErrors) override;
		void calculateGradients(const nnet_float *features) override;
		void updateWeights(const unsigned int batchSize) override;
		void updateBiases(const unsigned int batchSize) override;

	protected:
		std::size_t numInputChannels;
		std::size_t numOutputChannels;
		std::size_t kernelVolume;
		std::size_t frequencyVolume;
		std::size_t inputVolume;
		std::size_t outputVolume;
		nnet_float *padded;
		nnet_float *frequencyActivations;
		nnet_float *frequencyWeights;
		nnet_float *frequencyInputs;
		nnet_float *frequencyDeltaErrors;
		nnet_float *frequencyDeltaWeights;
		fftwf_plan forwardTransform;
		fftwf_plan backwardTransform;
		std::size_t tensorRank;
		std::size_t *inputDimensions;
		std::size_t *kernelDimensions;
		std::size_t *outputDimensions;
		nnet_float initWeight;
		X86ActivationFunction *activationFunction;
		X86UpdateRule *updateRule;
};

#endif
