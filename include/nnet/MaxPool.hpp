#ifndef _MAXPOOL_HPP_
#define _MAXPOOL_HPP_

#include "Layer.hpp"

class MaxPool : public Layer
{
	public:
		MaxPool(std::size_t rank, const std::size_t *inputDims, std::size_t chans, const std::size_t *poolDims);
		void initialise() override;
		void forward(const nnet_float *features) override;
		void backward(nnet_float *bpDeltaErrors) override;
		void calculateGradients(const nnet_float *features) override;

	protected:
		std::size_t *inputDimensions;
		std::size_t *poolDimensions;
		std::size_t *outputDimensions;
		std::size_t *inputIndices;
		std::size_t inputVolume;
		std::size_t outputVolume;
		std::size_t channels;
		std::size_t tensorRank;
};

#endif
