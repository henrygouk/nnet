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

class SlidingSoftmax : public ActivationFunction
{
	public:
		SlidingSoftmax(size_t numlbls, size_t stride);
		void operator()(nnet_float *activations, nnet_float *deltaActivations, std::size_t numActivations) const override;

	protected:
		size_t numLabels;
		size_t stride;
};

class RectifiedLinear : public ActivationFunction
{
	public:
		void operator()(nnet_float *activations, nnet_float *deltaActiations, std::size_t numActivation) const override;
};

#endif
