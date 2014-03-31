#ifndef _TYPES_H_
#define _TYPES_H_

#include <math.h>
#include <stddef.h>

#include <fftw3.h>

#define EXP expf

#define NO_UPDATE 0
#define SGD 1
#define MOMENTUM 2
#define L2_DECAY 4

typedef float nnet_float_t;

typedef enum
{
	NONE,
	RECTIFIED,
	LOGISTIC,
	SOFTMAX
} activation_function_t;

typedef enum
{
	SQUARED_ERROR
} loss_function_t;

typedef struct
{
	int algorithm;
	nnet_float_t learning_rate;
	nnet_float_t momentum_rate;
	nnet_float_t l2_decay_rate;
} update_rule_t;

typedef struct
{
	nnet_float_t *activations;
	nnet_float_t *errors;
	nnet_float_t *weights;
	nnet_float_t *gradients;
	size_t num_weights;
	size_t num_inputs;
	size_t num_units;
	struct layer_vtable *vtable;
	void *layer_data;
	update_rule_t *update_rule;
} layer_t;

struct layer_vtable
{
	void (* destroy)(layer_t *layer);
	void (* forward)(layer_t *layer, nnet_float_t *features, int train);
	void (* backward)(layer_t *layer, nnet_float_t *bperrs);
	void (* calculate_gradients)(layer_t *layer, nnet_float_t *features);
	void (* start_batch)(layer_t *layer);
	void (* end_batch)(layer_t *layer);
};

typedef struct layer_vtable layer_vtable_t;

void layer_destroy(layer_t *layer);
void layer_forward(layer_t *layer, nnet_float_t *features, int train);
void layer_calculate_activations(layer_t *layer, nnet_float_t *delta_activations, activation_function_t activation_function);
void layer_backward(layer_t *layer, nnet_float_t *bperrs);
void layer_calculate_gradients(layer_t *layer, nnet_float_t *features);
void layer_update(layer_t *layer);
void layer_start_batch(layer_t *layer);
void layer_end_batch(layer_t *layer);

#endif
