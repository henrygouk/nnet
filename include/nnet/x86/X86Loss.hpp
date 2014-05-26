#ifndef _LOSS_HPP_
#define _LOSS_HPP_

#include "../types.hpp"

class X86Loss
{
	public:
		virtual void loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaActivations, std::size_t length) = 0;
};

class X86SquaredError : public X86Loss
{
	public:
		void loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaActivations, std::size_t length) override;
};

class X86CrossEntropy : public X86Loss
{
	public:
		void loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaActivations, std::size_t length) override;
};

#endif
