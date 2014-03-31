#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <fftw3.h>

#include "core.h"
#include "fftconv.h"
#include "vector.h"
#include "conv.h"

layer_vtable_t fftconv_vtable;

layer_t *fftconv_create(size_t num_input_images, size_t input_dims, size_t num_output_images, size_t kernel_dims, activation_function_t func, nnet_float_t weight_size)
{
	layer_t *layer = (layer_t *)malloc(sizeof(layer_t));
	fftconv_layer_data_t *layer_data = (fftconv_layer_data_t *)malloc(sizeof(fftconv_layer_data_t));

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
	
	layer_data->delta_activations = nnet_malloc(layer->num_units);
	layer_data->padded = nnet_malloc(padded_dims * padded_dims);
	layer_data->frequency_activations = nnet_malloc(fsize * 2);
	layer_data->frequency_kernels = nnet_malloc(fsize * 2 * num_input_images * num_output_images);
	layer_data->frequency_gradients = nnet_malloc(fsize * 2 * num_input_images * num_output_images);
	layer_data->frequency_inputs = nnet_malloc(fsize * 2 * num_input_images);
	layer_data->frequency_errors = nnet_malloc(fsize * 2 * num_output_images);
	layer_data->frequency_size = fsize;
	layer_data->padded_dims = padded_dims;
	layer_data->input_dims = input_dims;
	layer_data->output_dims = output_dims;
	layer_data->kernel_dims = kernel_dims;
	layer_data->num_input_maps = num_input_images;
	layer_data->num_output_maps = num_output_images;
	layer_data->activation_function = func;

	//Randomly initialise the weights
	//TODO: let the user specify the range (also look into using a better distribution)
	random_vector(layer->weights, layer->num_weights, -weight_size, weight_size);

	//Set the gradients to 0
	memset(layer_data->frequency_gradients, 0, sizeof(nnet_float_t) * layer_data->frequency_size * 2 * num_input_images * num_output_images);
	memset(layer->gradients, 0, sizeof(nnet_float_t) * layer->num_weights);

	//Set pointers for the vtable
	fftconv_vtable.destroy = &fftconv_destroy;
	fftconv_vtable.forward = &fftconv_forward;
	fftconv_vtable.backward = &fftconv_backward;
	fftconv_vtable.calculate_gradients = &fftconv_calculate_gradients;
	fftconv_vtable.start_batch = &fftconv_start_batch;
	fftconv_vtable.end_batch = &fftconv_end_batch;

	//Create FFTW plans
	layer_data->forward = fftwf_plan_dft_r2c_2d(padded_dims, padded_dims, layer_data->padded, (fftwf_complex *)layer_data->frequency_inputs, FFTW_ESTIMATE);
	layer_data->backward = fftwf_plan_dft_c2r_2d(padded_dims, padded_dims, (fftwf_complex *)layer_data->frequency_activations, layer_data->padded, FFTW_ESTIMATE);

	layer->vtable = &fftconv_vtable;
	layer->layer_data = layer_data;
	
	return layer;
}

void fftconv_destroy(layer_t *layer)
{
	fftconv_layer_data_t *layer_data = layer->layer_data;
	nnet_free(layer->weights);
	nnet_free(layer->gradients);
	nnet_free(layer->activations);
	nnet_free(layer->errors);
	nnet_free(layer_data->delta_activations);
	nnet_free(layer_data->frequency_activations);
	nnet_free(layer_data->frequency_kernels);
	nnet_free(layer_data->frequency_gradients);
	nnet_free(layer_data->padded);
	nnet_free(layer_data->frequency_inputs);
	nnet_free(layer_data->frequency_errors);
	fftwf_destroy_plan(layer_data->forward);
	fftwf_destroy_plan(layer_data->backward);
	free(layer->layer_data);
	free(layer);
}

void fftconv_forward(layer_t *layer, nnet_float_t *inputs, int train)
{
	fftconv_layer_data_t *layer_data = layer->layer_data;
	nnet_float_t *finputs;
	nnet_float_t *fkernels = layer_data->frequency_kernels;
	nnet_float_t *factivations = layer_data->frequency_activations;
	nnet_float_t *biases = layer->weights + layer->num_weights - layer_data->num_output_maps;
	nnet_float_t *activations = layer->activations;

	nnet_float_t norm = 1.0 / (nnet_float_t)(layer_data->padded_dims * layer_data->padded_dims);

	//Compute FFT of each input map
	for(size_t i = 0; i < layer_data->num_input_maps; i++)
	{
		pad(inputs + i * layer_data->input_dims * layer_data->input_dims, layer_data->input_dims, layer_data->padded, layer_data->padded_dims);
		fftwf_execute_dft_r2c(layer_data->forward, layer_data->padded, (fftwf_complex *)(layer_data->frequency_inputs + i * layer_data->frequency_size * 2));
	}

	//Iterate over each output map
	for(size_t u = 0; u < layer_data->num_output_maps; u++)
	{
		finputs = layer_data->frequency_inputs;
		memset(factivations, 0, sizeof(nnet_float_t) * layer_data->frequency_size * 2);

		//Iterate over each input map
		for(size_t i = 0; i < layer_data->num_input_maps; i++)
		{
			vector_complex_fma(factivations, fkernels, finputs, layer_data->frequency_size);
			
			fkernels += layer_data->frequency_size * 2;
			finputs += layer_data->frequency_size * 2;
		}

		//Add the bias term to the DC component.. faster than doing it in the space domain ;)
		factivations[0] += biases[u] * (nnet_float_t)(layer_data->padded_dims * layer_data->padded_dims);

		//Inverse FFT
		fftwf_execute_dft_c2r(layer_data->backward, (fftwf_complex *)factivations, layer_data->padded);

		//Extract map from zero padded array
		extract_valid(layer_data->padded, layer_data->padded_dims, activations, layer_data->output_dims, layer_data->kernel_dims);

		//Normalise the inverse FFT
		vector_scale(activations, layer_data->output_dims * layer_data->output_dims, norm);

		activations += layer_data->output_dims * layer_data->output_dims;
	}
	
	layer_calculate_activations(layer, layer_data->delta_activations, layer_data->activation_function);
}

void fftconv_backward(layer_t *layer, nnet_float_t *bperrs)
{
	fftconv_layer_data_t *layer_data = layer->layer_data;
	nnet_float_t norm = 1.0 / (nnet_float_t)(layer_data->padded_dims * layer_data->padded_dims);
	nnet_float_t *ferrors = layer_data->frequency_errors;
	nnet_float_t *fkernels = layer_data->frequency_kernels;
	nnet_float_t *ftemp = layer_data->frequency_activations;

	for(size_t o = 0; o < layer_data->num_output_maps; o++)
	{
		pad_rotate(layer->errors + o * layer_data->output_dims * layer_data->output_dims, layer_data->output_dims, layer_data->padded, layer_data->padded_dims);
		fftwf_execute_dft_r2c(layer_data->forward, layer_data->padded, (fftwf_complex *)(layer_data->frequency_errors + o * layer_data->frequency_size * 2));
	}

	for(size_t i = 0; i < layer_data->num_input_maps; i++)
	{
		memset(ftemp, 0, sizeof(nnet_float_t) * layer_data->frequency_size * 2);
		fkernels = layer_data->frequency_kernels + i * layer_data->frequency_size * 2;

		for(size_t o = 0; o < layer_data->num_output_maps; o++)
		{
			vector_complex_fma(ftemp, fkernels, ferrors + o * layer_data->frequency_size * 2, layer_data->frequency_size);

			fkernels += layer_data->num_input_maps * layer_data->frequency_size * 2;
		}

		//Inverse FFT
		fftwf_execute_dft_c2r(layer_data->backward, (fftwf_complex *)ftemp, layer_data->padded);

		//Extract the required region
		extract_full_rotate(layer_data->padded, layer_data->padded_dims, bperrs, layer_data->input_dims);

		//Normalise the inverse FFT
		vector_scale(bperrs, layer_data->input_dims * layer_data->input_dims, norm);

		bperrs += layer_data->input_dims * layer_data->input_dims;
	}
}

void fftconv_calculate_gradients(layer_t *layer, nnet_float_t *inputs)
{
	fftconv_layer_data_t *layer_data = layer->layer_data;
	nnet_float_t *fgradients = layer_data->frequency_gradients;
	nnet_float_t *ferrors = layer_data->frequency_errors;
	nnet_float_t *errors = layer->errors;
	nnet_float_t *finputs;
	nnet_float_t *bias_gradients = layer->gradients + layer->num_weights - layer_data->num_output_maps;

	//Finish calculating the error w.r.t each unit in this layer
	vector_mul(layer->errors, layer_data->delta_activations, layer->errors, layer->num_units);

	//Iterate over each output map
	for(size_t o = 0; o < layer_data->num_output_maps; o++)
	{
		finputs = layer_data->frequency_inputs;

		//Calculate the bias gradient
		bias_gradients[o] += vector_sum(errors, layer_data->output_dims * layer_data->output_dims);

		pad_rotate(errors, layer_data->output_dims, layer_data->padded, layer_data->padded_dims);
		fftwf_execute_dft_r2c(layer_data->forward, layer_data->padded, (fftwf_complex *)ferrors);

		//Iterate over each input map
		for(size_t i = 0; i < layer_data->num_input_maps; i++)
		{
			vector_complex_fma(fgradients, ferrors, finputs, layer_data->frequency_size);

			finputs += layer_data->frequency_size * 2;
			fgradients += layer_data->frequency_size * 2;
		}

		errors += layer_data->output_dims * layer_data->output_dims;
	}
}

void fftconv_end_batch(layer_t *layer)
{
	fftconv_layer_data_t *layer_data = layer->layer_data;

	nnet_float_t norm = 1.0 / (nnet_float_t)(layer_data->padded_dims * layer_data->padded_dims);

	nnet_float_t *fgradients = layer_data->frequency_gradients;
	nnet_float_t *gradients = layer->gradients;

	//Iterate over each kernel
	for(size_t m = 0; m < layer_data->num_output_maps * layer_data->num_input_maps; m++)
	{
		//Transform the gradients for this kernel back into the space domain
		fftwf_execute_dft_c2r(layer_data->backward, (fftwf_complex *)fgradients, layer_data->padded);

		//Extract the valid section
		extract_valid_rotate(layer_data->padded, layer_data->padded_dims, gradients, layer_data->kernel_dims, layer_data->output_dims);

		//Set the frequency domain gradients for this kernel to 0
		memset(fgradients, 0, sizeof(nnet_float_t) * layer_data->frequency_size * 2);

		//Normalise the inverse FFT
		vector_scale(gradients, layer_data->kernel_dims * layer_data->kernel_dims, norm);

		fgradients += layer_data->frequency_size * 2;
		gradients += layer_data->kernel_dims * layer_data->kernel_dims;
	}
}

void fftconv_start_batch(layer_t *layer)
{
	fftconv_layer_data_t *layer_data = layer->layer_data;

 	//Transform the new FFT kernels back into the frequency domain
	for(size_t i = 0; i < layer_data->num_input_maps * layer_data->num_output_maps; i++)
	{
		pad(layer->weights + i * layer_data->kernel_dims * layer_data->kernel_dims, layer_data->kernel_dims, layer_data->padded, layer_data->padded_dims);
		fftwf_execute_dft_r2c(layer_data->forward, layer_data->padded, (fftwf_complex *)(layer_data->frequency_kernels + i * layer_data->frequency_size * 2));
	}
}

