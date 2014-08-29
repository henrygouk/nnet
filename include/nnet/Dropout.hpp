#ifndef _DROPOUT_HPP_
#define _DROPOUT_HPP_

#include "types.hpp"
#include "Layer.hpp"

class Dropout : public Layer
{
	public:
		Dropout(std::size_t numinputs, nnet_float prob);
		~Dropout();
		void forwardTrain(const nnet_float *features) override;
		void forward(const nnet_float *features) override;
		void backward(nnet_float *bpDeltaErrors) override;
		std::string toString() const override;

	protected:
		nnet_float dropoutProbability;
		nnet_float forwardScalar;
};

#endif
