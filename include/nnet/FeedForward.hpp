#ifndef _FEEDFORWARD_H_
#define _FEEDFORWARD_H_

#include <vector>

#include "Layer.hpp"
#include "Loss.hpp"
#include "types.hpp"

class FeedForward
{
	public:
		FeedForward(const std::vector<Layer *> &layervec, Loss *lf);
		virtual ~FeedForward();
		void save(std::ostream &os);
		void load(std::istream &is);
		void train(const nnet_float *features, const nnet_float *labels, const std::size_t numInstances, std::uint32_t epochs, std::uint32_t batchSize);
		void predict(const nnet_float *features, nnet_float *labels);
		std::string toString() const;
		size_t outputsSize() const;
		const std::vector<Layer *> &getLayers() const;

	private:
		std::uint32_t numFeatures;
		std::uint32_t numLabels;
		Loss *lossFunction;

		std::vector<Layer *> layers;
		nnet_float *hypothesis;

		void forward(const nnet_float *features);
		void backward(const nnet_float *features);
		void update(const unsigned int batchSize);
};

#endif
