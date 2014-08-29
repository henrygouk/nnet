#ifndef _LOSS_HPP_
#define _LOSS_HPP_

#include <string>

#include "types.hpp"

class Loss
{
	public:
		virtual void loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaActivations, std::size_t length) = 0;
		virtual nnet_float evaluate(const nnet_float *hypotheses, const nnet_float *labels, std::size_t length, std::size_t count) = 0;
		virtual std::string toString() const;
};

class SquaredError : public Loss
{
	public:
		void loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaActivations, std::size_t length) override;
		nnet_float evaluate(const nnet_float *hypotheses, const nnet_float *labels, std::size_t length, std::size_t count) override;
		std::string toString() const override;
};

class NegativeCosine : public Loss
{
	public:
		void loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaActivations, std::size_t length) override;
		nnet_float evaluate(const nnet_float *hypotheses, const nnet_float *labels, std::size_t length, std::size_t count) override;
};

class CrossEntropy : public Loss
{
	public:
		void loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaActivations, std::size_t length) override;
		nnet_float evaluate(const nnet_float *hypotheses, const nnet_float *labels, std::size_t length, std::size_t count) override;
		std::string toString() const override;
};

#endif
