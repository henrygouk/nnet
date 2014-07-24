#include "nnet/core.hpp"
#include "nnet/FeedForward.hpp"

nnet_float evaluate(FeedForward *ffnn, nnet_float *features, nnet_float *labels, size_t count, size_t num_features, size_t num_outputs)
{
	nnet_float *output = nnet_malloc(10);
	size_t correct = 0;

	for(size_t i = 0; i < count; i++)
	{
		ffnn->predict(features + i * num_features, output);

		size_t output_maxind = 0;
		nnet_float output_maxval = output[0];
		size_t labels_maxind = 0;
		nnet_float labels_maxval = labels[i * num_outputs];

		for(size_t j = 1; j < num_outputs; j++)
		{
			if(output[j] > output_maxval)
			{
				output_maxind = j;
				output_maxval = output[j];
			}

			if(labels[i * num_outputs + j] > labels_maxval)
			{
				labels_maxind = j;
				labels_maxval = labels[i * num_outputs + j];
			}
		}

		if(output_maxind == labels_maxind)
		{
			correct++;
		}
	}

	nnet_free(output);

	return (nnet_float)correct / (nnet_float)count;
}
