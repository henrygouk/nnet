#ifndef _UPDATERULE_HPP_
#define _UPDATERULE_HPP_

#include "X86Layer.hpp"
#include "../types.hpp"

class X86UpdateRule
{
	public:
		virtual void updateWeights(X86Layer *layer, const unsigned int batchSize) = 0;
		virtual void updateBiases(X86Layer *layer, const unsigned int batchSize) = 0;
};

class X86SGD : public X86UpdateRule
{
	public:
		void updateWeights(X86Layer *layer, const unsigned int batchSize) override;
		void updateBiases(X86Layer *layer, const unsigned int batchSize) override;
		nnet_float learningRate;
		nnet_float momentumRate;
		nnet_float l2DecayRate;
};

#endif
