#ifndef _FEEDFORWARD_H_
#define _FEEDFORWARD_H_

#include <vector>

#include "X86Layer.hpp"
#include "X86Loss.hpp"
#include "../types.hpp"

class X86FeedForward
{
	public:
		X86FeedForward(const std::vector<X86Layer *> &layervec, X86Loss *lf);
		virtual ~X86FeedForward();
		void train(const nnet_float *features, const nnet_float *labels, const std::size_t numInstances, std::uint32_t epochs, std::uint32_t batchSize);
		void predict(const nnet_float *features, nnet_float *labels);

	private:
		std::uint32_t numFeatures;
		std::uint32_t numLabels;
		std::size_t numWeights;
		std::size_t numBiases;
		std::size_t numActivations;
		X86Loss *lossFunction;

		std::vector<X86Layer *> layers;
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
		void update(const unsigned int batchSize);
};

#endif
