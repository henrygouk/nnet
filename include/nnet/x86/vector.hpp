#ifndef _VECTOR_HPP_
#define _VECTOR_HPP_

#include "../types.hpp"

nnet_float dot_product(nnet_float *vec1, nnet_float *vec2, std::size_t length);
nnet_float vector_sum(nnet_float *vec, std::size_t length);
void matrix_vector_mul(const nnet_float *matrix, std::size_t rows, std::size_t cols, const nnet_float *vecin, nnet_float *vecout);
void matrix_trans_vector_mul(nnet_float *matrix, std::size_t rows, std::size_t cols, nnet_float *vecin, nnet_float *vecout);
void vector_accum(nnet_float *vec1, nnet_float *vec2, std::size_t length);
void vector_mul(nnet_float *vec1, nnet_float *vec2, nnet_float *output, std::size_t length);
void vector_scale_accum(nnet_float *vec1, const nnet_float *vec2, nnet_float scalar, std::size_t length);
void vector_complex_fma(nnet_float *accum, const nnet_float *a, const nnet_float *b, const std::size_t length);
void vector_complex_conj_fma(nnet_float *accum, nnet_float *a, nnet_float *b, std::size_t length);
void vector_scale(nnet_float *vector, std::size_t length, nnet_float scalar);
void pad(std::size_t rank, const nnet_float *input, const std::size_t *input_dims, nnet_float *output, const std::size_t *output_dims);
void pad_rotate(std::size_t rank, const nnet_float *input, const std::size_t *input_dims, nnet_float *output, const std::size_t *output_dims);
void extract_valid(std::size_t rank, const nnet_float *input, const std::size_t *input_dims, nnet_float *output, const std::size_t *output_dims);
void extract_full(nnet_float *input, std::size_t input_dims, nnet_float *output, std::size_t output_dims);
void extract_valid_rotate(std::size_t rank, const nnet_float *input, const std::size_t *input_dims, nnet_float *output, const std::size_t *output_dims, const nnet_float normaliser);
void extract_full_rotate(std::size_t rank, const nnet_float *input, const std::size_t *input_dims, nnet_float *output, const std::size_t *output_dims);
void convolve_valid(nnet_float *image, std::size_t image_dims, nnet_float *kernel, std::size_t kernel_dims, nnet_float *outputs);
void correlate_valid(nnet_float *image, std::size_t image_dims, nnet_float *kernel, std::size_t kernel_dims, nnet_float *output);
void correlate_full(nnet_float *image, std::size_t image_dims, nnet_float *kernel, std::size_t kernel_dims, nnet_float *output);
void rotate_180(nnet_float *input, std::size_t dims, nnet_float *output);
void random_vector(nnet_float *vector, std::size_t length, nnet_float lower, nnet_float upper);
void random_gaussian_vector(nnet_float *vector, std::size_t length, nnet_float mean, nnet_float stddev);
void tensor_maxpool_nd(size_t rank, const nnet_float *input, const size_t *input_dims, const size_t *pool_dims, nnet_float *output, size_t *input_indices, size_t index);

#endif
