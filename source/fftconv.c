#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3.h>

#include "core.h"
#include "fftconv.h"
#include "vector.h"
#include "conv.h"
layer_vtable_t fftconv_vtable;

layer_t *fftconv_create(size_t num_input_images, size_t input_dims, size_t num_output_images, size_t kernel_dims, activation_function_t func)
{
	layer_t *layer = (layer_t *)malloc(sizeof(layer_t));

	size_t output_dims = input_dims - kernel_dims + 1;
	size_t padded_dims = pow(2, ceil(log2(input_dims)));
	size_t fsize = padded_dims * (padded_dims / 2 + 1);
	layer->num_inputs = num_input_images * input_dims * input_dims;
	layer->num_units = num_output_images * output_dims * output_dims;
	layer->num_weights = num_input_images * num_output_images * kernel_dims * kernel_dims + num_output_images;
	layer->weights = nnet_malloc(layer->num_weights);
	layer->gradients = nnet_malloc(layer->num_weights);
	layer->activations = nnet_malloc(layer->num_units);
	layer->errors = nnet_malloc(layer->num_units);
	layer->delta_activations = nnet_malloc(layer->num_units);
	layer->padded = nnet_malloc(padded_dims * padded_dims);
	layer->frequency_activations = nnet_malloc(fsize * 2);
	layer->frequency_kernels = nnet_malloc(fsize * 2 * num_input_images * num_output_images);
	layer->frequency_gradients = nnet_malloc(fsize * 2 * num_input_images * num_output_images);
	layer->frequency_inputs = nnet_malloc(fsize * 2 * num_input_images);
	layer->frequency_errors = nnet_malloc(fsize * 2 * num_output_images);
	layer->frequency_size = fsize;
	layer->padded_dims = padded_dims;
	layer->input_dims = input_dims;
	layer->output_dims = output_dims;
	layer->kernel_dims = kernel_dims;
	layer->num_input_maps = num_input_images;
	layer->num_output_maps = num_output_images;
	layer->activation_function = func;

	//Randomly initialise the weights
	//TODO: let the user specify the range (also look into using a better distribution)
	random_vector(layer->weights, layer->num_weights, -0.01, 0.01);

	//Set the gradients to 0
	memset(layer->frequency_gradients, 0, sizeof(nnet_float_t) * layer->frequency_size * 2 * num_input_images * num_output_images);
	memset(layer->gradients, 0, sizeof(nnet_float_t) * layer->num_weights);

	//Set pointers for the vtable
	fftconv_vtable.destroy = &fftconv_destroy;
	fftconv_vtable.forward = &fftconv_forward;
	fftconv_vtable.backward = &fftconv_backward;
	fftconv_vtable.calculate_gradients = &fftconv_calculate_gradients;
	fftconv_vtable.update = &fftconv_update;

	//Create FFTW plans
	layer->forward = fftwf_plan_dft_r2c_2d(padded_dims, padded_dims, layer->padded, (fftwf_complex *)layer->frequency_inputs, FFTW_ESTIMATE);
	layer->backward = fftwf_plan_dft_c2r_2d(padded_dims, padded_dims, (fftwf_complex *)layer->frequency_activations, layer->padded, FFTW_ESTIMATE);

	//Create frequency domain copies of the random weights
	for(size_t i = 0; i < num_input_images * num_output_images; i++)
	{
		pad(layer->weights + i * layer->kernel_dims * layer->kernel_dims, layer->kernel_dims, layer->padded, layer->padded_dims);
		fftwf_execute_dft_r2c(layer->forward, layer->padded, (fftwf_complex *)layer->frequency_kernels + i * layer->frequency_size);
	}

	layer->vtable = &fftconv_vtable;

	return layer;
}

void fftconv_destroy(layer_t *layer)
{
	nnet_free(layer->weights);
	nnet_free(layer->gradients);
	nnet_free(layer->activations);
	nnet_free(layer->errors);
	nnet_free(layer->delta_activations);
	nnet_free(layer->frequency_activations);
	nnet_free(layer->frequency_kernels);
	nnet_free(layer->frequency_gradients);
	nnet_free(layer->padded);
	nnet_free(layer->frequency_inputs);
	nnet_free(layer->frequency_errors);
	free(layer);
}

void fftconv_forward(layer_t *layer, nnet_float_t *inputs)
{
	nnet_float_t *finputs;
	nnet_float_t *fkernels = layer->frequency_kernels;
	nnet_float_t *factivations = layer->frequency_activations;
	nnet_float_t *biases = layer->weights + layer->num_weights - layer->num_output_maps;
	nnet_float_t *activations = layer->activations;

	nnet_float_t norm = 1.0 / (nnet_float_t)(layer->padded_dims * layer->padded_dims);

	//Compute FFT of each input map
	for(size_t i = 0; i < layer->num_input_maps; i++)
	{
		pad(inputs + i * layer->input_dims * layer->input_dims, layer->input_dims, layer->padded, layer->padded_dims);
		fftwf_execute_dft_r2c(layer->forward, layer->padded, (fftwf_complex *)layer->frequency_inputs + i * layer->frequency_size);
	}

	//Iterate over each output map
	for(size_t u = 0; u < layer->num_output_maps; u++)
	{
		finputs = layer->frequency_inputs;
		memset(factivations, 0, sizeof(nnet_float_t) * layer->frequency_size * 2);

		//Iterate over each input map
		for(size_t i = 0; i < layer->num_input_maps; i++)
		{
			vector_complex_fma(factivations, fkernels, finputs, layer->frequency_size);
			
			fkernels += layer->frequency_size * 2;
			finputs += layer->frequency_size * 2;
		}

		//Add the bias term to the DC component.. faster than doing it in the space domain ;)
		factivations[0] += biases[u] * (nnet_float_t)(layer->output_dims * layer->output_dims);

		//Inverse FFT
		fftwf_execute_dft_c2r(layer->backward, (fftwf_complex *)factivations, layer->padded);

		//Extract map from zero padded array
		extract_valid(layer->padded, layer->padded_dims, activations, layer->output_dims, layer->kernel_dims);

		//Normalise the inverse FFT
		vector_scale(activations, layer->output_dims * layer->output_dims, norm);

		activations += layer->output_dims * layer->output_dims;
	}

	layer_calculate_activations(layer);
}

void fftconv_backward(layer_t *layer, nnet_float_t *bperrs)
{
	nnet_float_t norm = 1.0 / (nnet_float_t)(layer->padded_dims * layer->padded_dims);
	nnet_float_t *ferrors = layer->frequency_errors;
	nnet_float_t *fkernels = layer->frequency_kernels;
	nnet_float_t *ftemp = layer->frequency_activations;

	for(size_t o = 0; o < layer->num_output_maps; o++)
	{
		pad(layer->errors + o * layer->output_dims * layer->output_dims, layer->output_dims, layer->padded, layer->padded_dims);
		fftwf_execute_dft_r2c(layer->forward, layer->padded, (fftwf_complex *)layer->frequency_errors + o * layer->frequency_size);
	}

	for(size_t i = 0; i < layer->num_input_maps; i++)
	{
		memset(ftemp, 0, sizeof(nnet_float_t) * layer->frequency_size * 2);

		for(size_t o = 0; o < layer->num_output_maps; o++)
		{
			vector_complex_conj_fma(ftemp, fkernels, ferrors, layer->frequency_size);

			fkernels += layer->frequency_size * 2;
		}

		//Inverse FFT
		fftwf_execute_dft_c2r(layer->backward, (fftwf_complex *)ftemp, layer->padded);

		//Extract the required region
		extract_full(layer->padded, layer->padded_dims, bperrs, layer->input_dims);

		//Normalise the inverse FFT
		vector_scale(bperrs, layer->input_dims * layer->input_dims, norm);

		ferrors += layer->frequency_size * 2;
		bperrs += layer->input_dims * layer->input_dims;
	}
}

void fftconv_calculate_gradients(layer_t *layer, nnet_float_t *inputs)
{
	nnet_float_t *fgradients = layer->frequency_gradients;
	nnet_float_t *ferrors = layer->frequency_errors;
	nnet_float_t *errors = layer->errors;
	nnet_float_t *finputs;
	nnet_float_t *bias_gradients = layer->gradients + layer->num_weights - layer->num_output_maps;

	//Finish calculating the error w.r.t each unit in this layer
	vector_mul(layer->errors, layer->delta_activations, layer->errors, layer->num_units);

	//Iterate over each output map
	for(size_t o = 0; o < layer->num_output_maps; o++)
	{
		finputs = layer->frequency_inputs;

		//Calculate the bias gradient
		bias_gradients[o] += vector_sum(errors, layer->output_dims * layer->output_dims);

		pad(errors, layer->output_dims, layer->padded, layer->padded_dims);
		fftwf_execute_dft_r2c(layer->forward, layer->padded, (fftwf_complex *)ferrors);

		//Iterate over each input map
		for(size_t i = 0; i < layer->num_input_maps; i++)
		{
			vector_complex_conj_fma(fgradients, ferrors, finputs, layer->frequency_size);

			finputs += layer->frequency_size * 2;
			fgradients += layer->frequency_size * 2;
		}

		errors += layer->output_dims * layer->output_dims;
	}
}

void fftconv_update(layer_t *layer, update_rule_t *update_rule)
{
	nnet_float_t norm = 1.0 / (nnet_float_t)(layer->padded_dims * layer->padded_dims);

	nnet_float_t *fgradients = layer->frequency_gradients;
	nnet_float_t *gradients = layer->gradients;

	//Iterate over each kernel
	for(size_t m = 0; m < layer->num_output_maps * layer->num_input_maps; m++)
	{
		//Transform the gradients for this kernel back into the space domain
		fftwf_execute_dft_c2r(layer->backward, (fftwf_complex *)fgradients, layer->padded);

		//Extract the valid section
		extract_valid(layer->padded, layer->padded_dims, gradients, layer->kernel_dims, layer->output_dims);

		//Set the frequency domain gradients for this kernel to 0
		memset(fgradients, 0, sizeof(nnet_float_t) * layer->frequency_size * 2);

		//Normalise the inverse FFT
		vector_scale(gradients, layer->kernel_dims * layer->kernel_dims, norm);

		fgradients += layer->frequency_size * 2;
		gradients += layer->kernel_dims * layer->kernel_dims;
	}

	//Do the weight update
	for(size_t w = 0; w < layer->num_weights; w++)
	{
		layer->weights[w] -= layer->gradients[w] * update_rule->learning_rate;
		layer->gradients[w] = 0.0;
	}
	
	//Transform the new FFT kernels back into the frequency domain
	for(size_t i = 0; i < layer->num_input_maps * layer->num_output_maps; i++)
	{
		pad(layer->weights + i * layer->kernel_dims * layer->kernel_dims, layer->kernel_dims, layer->padded, layer->padded_dims);
		fftwf_execute_dft_r2c(layer->forward, layer->padded, (fftwf_complex *)layer->frequency_kernels + i * layer->frequency_size);
	}
}
