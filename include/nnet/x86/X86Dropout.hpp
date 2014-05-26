#ifndef _X86DROPOUT_HPP_
#define _X86DROPOUT_HPP_

#include "../types.hpp"
#include "X86Layer.hpp"

class X86Dropout : public X86Layer
{
	public:
		X86Dropout(std::size_t numinputs, nnet_float prob);
		void initialise() override;
		void forwardTrain(const nnet_float *features) override;
		void forward(const nnet_float *features) override;
		void backward(nnet_float *bpDeltaErrors) override;

	protected:
		nnet_float dropoutProbability;
		nnet_float forwardScalar;
};

#endif
