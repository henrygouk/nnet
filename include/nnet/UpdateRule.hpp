#ifndef _UPDATERULE_HPP_
#define _UPDATERULE_HPP_

#include "Layer.hpp"
#include "types.hpp"

class UpdateRule
{
	public:
		virtual void updateWeights(Layer *layer, const unsigned int batchSize) = 0;
		virtual void updateBiases(Layer *layer, const unsigned int batchSize) = 0;
};

class SGD : public UpdateRule
{
	public:
		void updateWeights(Layer *layer, const unsigned int batchSize) override;
		void updateBiases(Layer *layer, const unsigned int batchSize) override;
		nnet_float learningRate;
		nnet_float momentumRate;
		nnet_float l2DecayRate;
};

#endif
