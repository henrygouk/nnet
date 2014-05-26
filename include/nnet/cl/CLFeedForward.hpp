#ifndef _CLFEEDFORWARD_HPP_
#define _CLFEEDFORWARD_HPP_

class CLFeedForward
{
	public:
		CLFeedForward(const std::vector<CLLayer *>, CLLoss *lf);
		virtual ~CLFeedForward();
		void train(const nnet_float *features, const nnet_float *labels, const std::size_t numInstances, std::uint32_t epochs, std::uint32_t batchSize);
		void predict(const nnet_float *features, nnet_float *labels);

	private:
		std::uint32_t numFeatures;
		std::uint32_t numLabels;
		std::size_t numWeights;
		std::size_t numBiases;
		std::size_t numActivations;

};

#endif
