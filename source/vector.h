#ifndef _LINALG_H_
#define _LINALG_H_

#include "types.h"

nnet_float_t dot_product(nnet_float_t *vec1, nnet_float_t *vec2, size_t length);
nnet_float_t vector_sum(nnet_float_t *vec, size_t length);
void matrix_vector_mul(nnet_float_t *matrix, size_t rows, size_t cols, nnet_float_t *vecin, nnet_float_t *vecout);
void matrix_trans_vector_mul(nnet_float_t *matrix, size_t rows, size_t cols, nnet_float_t *vecin, nnet_float_t *vecout);
void vector_accum(nnet_float_t *vec1, nnet_float_t *vec2, size_t length);
void vector_mul(nnet_float_t *vec1, nnet_float_t *vec2, nnet_float_t *output, size_t length);
void vector_scale_accum(nnet_float_t *vec1, nnet_float_t *vec2, nnet_float_t scalar, size_t length);
void vector_complex_fma(nnet_float_t *accum, nnet_float_t *a, nnet_float_t *b, size_t length);
void vector_complex_conj_fma(nnet_float_t *accum, nnet_float_t *a, nnet_float_t *b, size_t length);
void vector_scale(nnet_float_t *vector, size_t length, nnet_float_t scalar);
void pad(nnet_float_t * input, size_t input_dims, nnet_float_t *output, size_t output_dims);
void extract_valid(nnet_float_t *input, size_t input_dims, nnet_float_t *output, size_t output_dims, size_t kernel_dims);
void extract_full(nnet_float_t *input, size_t input_dims, nnet_float_t *output, size_t output_dims);
void convolve_valid(nnet_float_t *image, size_t image_dims, nnet_float_t *kernel, size_t kernel_dims, nnet_float_t *outputs);
void correlate_valid(nnet_float_t *image, size_t image_dims, nnet_float_t *kernel, size_t kernel_dims, nnet_float_t *output);
void correlate_full(nnet_float_t *image, size_t image_dims, nnet_float_t *kernel, size_t kernel_dims, nnet_float_t *output);

#endif
