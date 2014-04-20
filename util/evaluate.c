#include "../source/core.h"
#include "../source/ffnn.h"
#include "../source/types.h"

nnet_float_t evaluate(ffnn_t *ffnn, nnet_float_t *features, nnet_float_t *labels, size_t count, size_t num_features, size_t num_outputs)
{
	nnet_float_t *output = nnet_malloc(10);
	size_t correct = 0;

	for(size_t i = 0; i < count; i++)
	{
		ffnn_predict(ffnn, features + i * num_features, output);

		size_t output_maxind = 0;
		nnet_float_t output_maxval = output[0];
		size_t labels_maxind = 0;
		nnet_float_t labels_maxval = labels[i * num_outputs];

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

	return (nnet_float_t)correct / (nnet_float_t)count;
}
