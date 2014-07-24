#ifndef _ACTIVATIONFUNCTION_HPP_
#define _ACTIVATIONFUNCTION_HPP_

#include "types.hpp"

class ActivationFunction
{
	public:
		virtual void operator()(nnet_float *activations, nnet_float *deltaActiations, std::size_t numActivations) const = 0;
};

class Logistic : public ActivationFunction
{
	public:
		void operator()(nnet_float *activations, nnet_float *deltaActiations, std::size_t numActivation) const override;
};

class Softmax : public ActivationFunction
{
	public:
		void operator()(nnet_float *activations, nnet_float *deltaActiations, std::size_t numActivation) const override;
};

class RectifiedLinear : public ActivationFunction
{
	public:
		void operator()(nnet_float *activations, nnet_float *deltaActiations, std::size_t numActivation) const override;
};

#endif
