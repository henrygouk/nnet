#ifndef _SPATIALCONVOLUTIONAL_HPP_
#define _SPATIALCONVOLUTIONAL_HPP_

#include "ActivationFunction.hpp"
#include "Layer.hpp"
#include "UpdateRule.hpp"

class SpatialConvolutional : public Layer
{
	public:
		SpatialConvolutional(const std::size_t *inputDims, const std::size_t *kernelDims, std::size_t inputs, std::size_t outputs, nnet_float initweight, ActivationFunction *func, UpdateRule *ur);
		~SpatialConvolutional();
		void endBatch() override;
		void forward(const nnet_float *features) override;
		void backward(nnet_float *bpDeltaErrors) override;
		void calculateGradients(const nnet_float *features) override;
		void updateWeights(const unsigned int batchSize) override;
		void updateBiases(const unsigned int batchSize) override;

	protected:
		std::size_t *inputDims;
		std::size_t *kernelDims;
		std::size_t *outputDims;
		std::size_t inputVolume;
		std::size_t outputVolume;
		std::size_t kernelVolume;
		std::size_t numInputChannels;
		std::size_t numOutputChannels;
		nnet_float initWeight;
		ActivationFunction *activationFunction;
		UpdateRule *updateRule;

};

#endif
