#ifndef _FEEDFORWARD_H_
#define _FEEDFORWARD_H_

#include <vector>

#include "Layer.hpp"
#include "types.hpp"

class FeedForward
{
	public:
		FeedForward(const std::vector<Layer *> &layervec, LossFunction lf);
		~FeedForward();
		void train(const nnet_float *features, const nnet_float *labels, const std::size_t numInstances, std::uint32_t epochs, std::uint32_t batchSize);
		void predict(const nnet_float *features, nnet_float *labels);

	private:
		std::uint32_t numFeatures;
		std::uint32_t numLabels;
		std::size_t numWeights;
		std::size_t numBiases;
		std::size_t numActivations;
		LossFunction lossFunction;

		std::vector<Layer *> layers;
		nnet_float *hypothesis;
		nnet_float *weights;
		nnet_float *deltaWeights;
		nnet_float *biases;
		nnet_float *deltaBiases;
		nnet_float *activations;
		nnet_float *deltaActivations;
		nnet_float *deltaErrors;

		void forward(const nnet_float *features);
		void backward(const nnet_float *features, const nnet_float *deltaErrors);
		void update();
};

#endif
