#ifndef _LOSS_HPP_
#define _LOSS_HPP_

#include <string>

#include "types.hpp"

class Loss
{
	public:
		virtual void loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaActivations, std::size_t length) = 0;
		virtual std::string toString() const;
};

class SquaredError : public Loss
{
	public:
		void loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaActivations, std::size_t length) override;
		std::string toString() const override;
};

class CrossEntropy : public Loss
{
	public:
		void loss(const nnet_float *hypothesis, const nnet_float *labels, nnet_float *deltaActivations, std::size_t length) override;
		std::string toString() const override;
};

#endif
