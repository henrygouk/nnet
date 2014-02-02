#ifndef _TYPES_H_
#define _TYPES_H_

#include <stddef.h>

#include <fftw3.h>

#define EXP expf

typedef float nnet_float_t;

typedef enum
{
	NONE,
	RECTIFIED,
	LOGISTIC
} activation_function_t;

typedef struct
{
	nnet_float_t *activations;
	nnet_float_t *frequency_activations;
	nnet_float_t *frequency_kernels;
	nnet_float_t *frequency_inputs;
	nnet_float_t *frequency_errors;
	nnet_float_t *frequency_gradients;
	nnet_float_t *padded;
	nnet_float_t *delta_activations;
	nnet_float_t *errors;
	nnet_float_t *weights;
	nnet_float_t *gradients;
	size_t padded_dims;
	size_t frequency_size;
	size_t num_inputs;
	size_t num_units;
	size_t num_weights;
	size_t input_dims;
	size_t output_dims;
	size_t kernel_dims;
	size_t num_input_maps;
	size_t num_output_maps;
	struct layer_vtable *vtable;
	activation_function_t activation_function;
	fftwf_plan forward, backward;
} layer_t;

typedef struct
{
	nnet_float_t learning_rate;
} update_rule_t;

struct layer_vtable
{
	void (* destroy)(layer_t *layer);
	void (* forward)(layer_t *layer, nnet_float_t *features);
	void (* backward)(layer_t *layer, nnet_float_t *bperrs);
	void (* calculate_gradients)(layer_t *layer, nnet_float_t *features);
	void (* update)(layer_t *layer, update_rule_t *learning_rate);
};

typedef struct layer_vtable layer_vtable_t;

void layer_destroy(layer_t *layer);
void layer_forward(layer_t *layer, nnet_float_t *features);
void layer_calculate_activations(layer_t *layer);
void layer_backward(layer_t *layer, nnet_float_t *bperrs);
void layer_calculate_gradients(layer_t *layer, nnet_float_t *features);
void layer_update(layer_t *layer, update_rule_t *learning_rate);

#endif
