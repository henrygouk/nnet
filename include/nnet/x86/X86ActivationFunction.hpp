#ifndef _ACTIVATIONFUNCTION_HPP_
#define _ACTIVATIONFUNCTION_HPP_

#include "../types.hpp"

class X86ActivationFunction
{
	public:
		virtual void operator()(nnet_float *activations, nnet_float *deltaActiations, std::size_t numActivations) const = 0;
};

class X86Logistic : public X86ActivationFunction
{
	public:
		void operator()(nnet_float *activations, nnet_float *deltaActiations, std::size_t numActivation) const override;
};

class X86Softmax : public X86ActivationFunction
{
	public:
		void operator()(nnet_float *activations, nnet_float *deltaActiations, std::size_t numActivation) const override;
};

class X86RectifiedLinear : public X86ActivationFunction
{
	public:
		void operator()(nnet_float *activations, nnet_float *deltaActiations, std::size_t numActivation) const override;
};

#endif
