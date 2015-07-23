#include <cstdlib>
#include <cstring>
#include <random>

#include "nnet/vector.hpp"

using namespace std;

#ifdef HAVE_AVX

#include <x86intrin.h>

#define VECTOR __m256
#define VECTOR_LENGTH 8
#define VADD _mm256_add_ps
#define VSUB _mm256_sub_ps
#define VMUL _mm256_mul_ps
#define VDIV _mm256_div_ps
#define VSETZERO _mm256_setzero_ps
#define VSET1 _mm256_set1_ps
#define VSET _mm256_set_ps
#define VLOAD _mm256_loadu_ps
#define VLOADU _mm256_loadu_ps
#define VSTORE _mm256_storeu_ps
#define VSTOREU _mm256_storeu_ps

static inline nnet_float VHADD(VECTOR vec)
{
	VECTOR vec2 = _mm256_permute2f128_ps(vec, vec, 1);
	vec = _mm256_add_ps(vec, vec2);
	vec = _mm256_hadd_ps(vec, vec);
	vec = _mm256_hadd_ps(vec, vec);
	return _mm_cvtss_f32(_mm256_castps256_ps128(vec));
}

static inline VECTOR VCMUL(VECTOR a, VECTOR b)
{
	__m256 b_flip = _mm256_shuffle_ps(b,b,0xB1);
	__m256 a_im = _mm256_shuffle_ps(a,a,0xF5);
	__m256 a_re = _mm256_shuffle_ps(a,a,0xA0);
	__m256 aib = _mm256_mul_ps(a_im, b_flip);
	__m256 arb = _mm256_mul_ps(a_re, b);
	return _mm256_addsub_ps(arb, aib);
}

static inline VECTOR VCFMA(VECTOR c, VECTOR a, VECTOR b)
{
	__m256 b_flip = _mm256_shuffle_ps(b,b,0xB1);
	__m256 a_re = _mm256_shuffle_ps(a,a,0xA0);
	__m256 a_im = _mm256_shuffle_ps(a,a,0xF5);

#ifdef HAVE_FMA
	__m256 arb = _mm256_fmadd_ps(a_re, b, c);
#else
	__m256 arb = _mm256_add_ps(_mm256_mul_ps(a_re, b), c);
#endif
	
	__m256 aib = _mm256_mul_ps(a_im, b_flip);
	return _mm256_addsub_ps(arb, aib);
}

#else

//Fall back to scalar arithmetic

#define VECTOR nnet_float
#define VECTOR_LENGTH 1
#define VADD(a, b) (a + b)
#define VSUB(a, b) (a - b)
#define VMUL(a, b) (a * b)
#define VDIV(a, b) (a / b)
#define VHADD(a) (a)
#define VSETZERO() (0)
#define VSET(a) (a)
#define VSET1(a) (a)
#define VLOAD(addr) (*(addr))
#define VLOADU(addr) (*(addr))
#define VSTORE(addr, a) (*(addr) = a)
#define VSTOREU(addr, a) (*(addr) = a)

#endif

#define VECTOR_FOR(COUNTER, LENGTH) for(COUNTER = 0; COUNTER <= LENGTH - VECTOR_LENGTH && LENGTH >= VECTOR_LENGTH; COUNTER += VECTOR_LENGTH)

nnet_float dot_product(const nnet_float *vec1, const nnet_float *vec2, size_t length)
{
	nnet_float accum = 0;
	VECTOR vecaccum = VSETZERO();
	size_t i = 0;

	VECTOR_FOR(i, length)
	{
		vecaccum = VADD(vecaccum, VMUL(VLOADU(vec1 + i), VLOADU(vec2 + i)));
	}

	accum = VHADD(vecaccum);

	for(; i < length; i++)
	{
		accum += vec1[i] * vec2[i];
	}

	return accum;
}

nnet_float vector_sum(nnet_float *vec, size_t length)
{
	size_t i = 0;
	nnet_float res = 0;
	VECTOR vecres = VSETZERO();

	VECTOR_FOR(i, length)
	{
		vecres = VADD(vecres, VLOADU(vec + i));
	}

	res = VHADD(vecres);

	for(; i < length; i++)
	{
		res += vec[i];
	}

	return res;
}

void matrix_vector_mul(const nnet_float *matrix, size_t rows, size_t cols, const nnet_float *vecin, nnet_float *vecout)
{
	for(size_t r = 0; r < rows; r++)
	{
		vecout[r] = dot_product(vecin, matrix + r * cols, cols);
	}
}

void matrix_trans_vector_mul(nnet_float *matrix, size_t rows, size_t cols, nnet_float *vecin, nnet_float *vecout)
{
	for(size_t c = 0; c < cols; c++)
	{
		vecout[c] = 0.0;
	}

	for(size_t r = 0; r < rows; r++)
	{
		size_t c = 0;
		VECTOR vin = VSET1(vecin[r]);

		VECTOR_FOR(c, cols)
		{
			VSTOREU(vecout + c, VADD(VLOADU(vecout + c), VMUL(VLOADU(matrix + r * cols + c), vin)));
		}

		for(; c < cols; c++)
		{
			vecout[c] += matrix[r * cols + c] * vecin[r];
		}
	}
}

void vector_accum(nnet_float *vec1, nnet_float *vec2, size_t length)
{
	size_t i = 0;

	for(; i < length; i++)
	{
		vec1[i] += vec2[i];
	}
}

void vector_mul(nnet_float *vec1, nnet_float *vec2, nnet_float *output, size_t length)
{
	size_t i = 0;

	VECTOR_FOR(i, length)
	{
		VSTOREU(output + i, VMUL(VLOADU(vec1 + i), VLOADU(vec2 + i)));
	}

	for(; i < length; i++)
	{
		output[i] = vec1[i] * vec2[i];
	}
}

void vector_scale_accum(nnet_float *vec1, const nnet_float *vec2, nnet_float scalar, size_t length)
{
	size_t i = 0;

	VECTOR vecscale = VSET1(scalar);

	VECTOR_FOR(i, length)
	{
		VSTOREU(vec1 + i, VADD(VLOADU(vec1 + i), VMUL(VLOADU(vec2 + i), vecscale)));
	}

	for(; i < length; i++)
	{
		vec1[i] += vec2[i] * scalar;
	}
}

void vector_complex_fma(nnet_float *accum, const nnet_float *a, const nnet_float *b, const size_t length)
{
	size_t i = 0;

#if HAVE_AVX

	for(; i <= length - (VECTOR_LENGTH >> 1) && length >= (VECTOR_LENGTH >> 1); i += (VECTOR_LENGTH >> 1))
	{
		VSTORE(accum, VCFMA(VLOAD(accum), VLOAD(a), VLOAD(b)));
		accum += VECTOR_LENGTH;
		a += VECTOR_LENGTH;
		b += VECTOR_LENGTH;
	}

#endif

	for(; i < length; i++)
	{
		accum[0] += a[0] * b[0] - a[1] * b[1];
		accum[1] += a[1] * b[0] + a[0] * b[1];
		accum += 2;
		a += 2;
		b += 2;
	}
}

void vector_complex_conj_fma(nnet_float *accum, nnet_float *a, nnet_float *b, size_t length)
{
	for(size_t i = 0; i < length; i++)
	{
		accum[0] += a[0] * b[0] + a[1] * b[1];
		accum[1] += a[1] * b[0] - a[0] * b[1];
		accum += 2;
		a += 2;
		b += 2;
	}
}

void vector_scale(nnet_float *vector, size_t length, nnet_float scalar)
{
	size_t i = 0;

	VECTOR vecscalar = VSET1(scalar);

	VECTOR_FOR(i, length)
	{
		VSTOREU(vector + i, VMUL(VLOADU(vector + i), vecscalar));
	}

	for(; i < length; i++)
	{
		vector[i] *= scalar;
	}
}

void tensor_maxpool_1d(const nnet_float *input, const size_t input_dims, const size_t pool_dims, nnet_float *output, const size_t output_dims, size_t *input_indices, size_t index)
{
	size_t i = 0;

	for(size_t p = 0; p < output_dims; p++)
	{
		for(size_t j = 0; j < pool_dims; j++, i++)
		{
			if(input[i] > output[p])
			{
				output[p] = input[i];
				input_indices[p] = index + i;
			}
		}
	}
}

void tensor_maxpool_nd(size_t rank, const nnet_float *input, const size_t *input_dims, const size_t *pool_dims, nnet_float *output, size_t *input_indices, size_t index)
{
	size_t input_volume = 1;
	size_t output_volume = 1;

	if(rank == 2)
	{
		size_t i = 0;
		const size_t output_dims0 = input_dims[0] / pool_dims[0];
		const size_t output_dims1 = input_dims[1] / pool_dims[1];

		for(size_t p = 0; p < output_dims0; p++)
		{
			for(size_t j = 0; j < pool_dims[0]; j++, i++)
			{
				tensor_maxpool_1d(input + i * input_dims[1], input_dims[1], pool_dims[1], output + p * output_dims1, output_dims1, input_indices + p * output_dims1, i * input_dims[1] + index);
			}
		}

		return;
	}
	else if(rank == 1)
	{
		tensor_maxpool_1d(input, input_dims[0], pool_dims[0], output, input_dims[0] / pool_dims[0], input_indices, index);
		return;
	}
	else if(rank == 0)
	{
		if(*output < *input)
		{
			*output = *input;
			*input_indices = index;
		}

		return;
	}

	for(size_t i = 1; i < rank; i++)
	{
		input_volume *= input_dims[i];
		output_volume *= (input_dims[i] / pool_dims[i]);
	}

	size_t i = 0;
	for(size_t p = 0; p < input_dims[0] / pool_dims[0]; p++)
	{
		for(size_t j = 0; j < pool_dims[0]; j++, i++)
		{
			tensor_maxpool_nd(rank - 1, input + i * input_volume, input_dims + 1, pool_dims + 1, output + p * output_volume, input_indices + p * output_volume, i * input_volume + index);
		}
	}
}

void pad(size_t rank, const nnet_float *input, const size_t *input_dims, nnet_float *output, const size_t *output_dims)
{
	size_t input_volume = 1;
	size_t output_volume = 1;
	size_t i;

	if(rank == 0)
	{
		*output = *input;
		return;
	}

	for(i = 1; i < rank; i++)
	{
		input_volume *= input_dims[i];
		output_volume *= output_dims[i];
	}

	for(i = 0; i < input_dims[0]; i++)
	{
		pad(rank - 1, input + i * input_volume, input_dims + 1, output + i * output_volume, output_dims + 1);
	}

	memset(output + i * output_volume, 0, sizeof(nnet_float) * output_volume * (output_dims[0] - input_dims[0]));
}

void pad_rotate_1d(const nnet_float *input, const size_t input_dims, nnet_float *output, const size_t output_dims)
{
	for(size_t i = 0; i < input_dims; i++)
	{
		output[i] = input[input_dims - i - 1];
	}

	memset(output + input_dims, 0, sizeof(nnet_float) * (output_dims - input_dims));
}

void pad_rotate(size_t rank, const nnet_float *input, const size_t *input_dims, nnet_float *output, const size_t *output_dims)
{
	size_t input_volume = 1;
	size_t output_volume = 1;
	size_t i;

	if(rank == 1)
	{
		pad_rotate_1d(input, input_dims[0], output, output_dims[0]);
		return;
	}
	else if(rank == 0)
	{
		*output = *input;
		return;
	}

	for(i = 1; i < rank; i++)
	{
		input_volume *= input_dims[i];
		output_volume *= output_dims[i];
	}

	for(i = 0; i < input_dims[0]; i++)
	{
		pad_rotate(rank - 1, input + (input_dims[0] - i - 1) * input_volume, input_dims + 1, output + i * output_volume, output_dims + 1);
	}

	memset(output + i * output_volume, 0, sizeof(nnet_float) * output_volume * (output_dims[0] - input_dims[0]));
}

void extract_valid_1d(const nnet_float *input, const size_t input_dims, nnet_float *output, const size_t output_dims)
{
	const size_t offset = input_dims - output_dims;

	for(size_t i = 0; i < output_dims; i++)
	{
		output[i] = input[i + offset];
	}
}

void extract_valid(size_t rank, const nnet_float *input, const size_t *input_dims, nnet_float *output, const size_t *output_dims)
{
	size_t input_volume = 1;
	size_t output_volume = 1;
	size_t i;

	if(rank == 1)
	{
		extract_valid_1d(input, input_dims[0], output, output_dims[0]);
		return;
	}
	else if(rank == 0)
	{
		*output = *input;
		return;
	}

	for(i = 1; i < rank; i++)
	{
		input_volume *= input_dims[i];
		output_volume *= output_dims[i];
	}

	size_t offset = input_dims[0] - output_dims[0];

	for(i = 0; i < output_dims[0]; i++)
	{
		extract_valid(rank - 1, input + (i + offset) * input_volume, input_dims + 1, output + i * output_volume, output_dims + 1);
	}
}

void extract_valid_rotate(size_t rank, const nnet_float *input, const size_t *input_dims, nnet_float *output, const size_t *output_dims, const nnet_float normaliser)
{
	size_t input_volume = 1;
	size_t output_volume = 1;
	size_t i;

	if(rank == 0)
	{
		*output += *input * normaliser;
		return;
	}

	for(i = 1; i < rank; i++)
	{
		input_volume *= input_dims[i];
		output_volume *= output_dims[i];
	}

	size_t offset = input_dims[0] - output_dims[0];

	for(i = 0; i < output_dims[0]; i++)
	{
		extract_valid_rotate(rank - 1, input + (i + offset) * input_volume, input_dims + 1, output + (output_dims[0] - i - 1) * output_volume, output_dims + 1, normaliser);
	}
}

void extract_full_rotate(size_t rank, const nnet_float *input, const size_t *input_dims, nnet_float *output, const size_t *output_dims)
{
	if(rank == 0)
	{
		*output = *input;
		return;
	}

	size_t input_volume = 1;
	size_t output_volume = 1;
	size_t i;

	for(i = 1; i < rank; i++)
	{
		input_volume *= input_dims[i];
		output_volume *= output_dims[i];
	}

	for(i = 0; i < output_dims[0]; i++)
	{
		extract_full_rotate(rank - 1, input + i * input_volume, input_dims + 1, output + (output_dims[0] - i - 1) * output_volume, output_dims + 1);
	}
}

void extract_full(nnet_float *input, size_t input_dims, nnet_float *output, size_t output_dims)
{
	for(size_t y = 0; y < output_dims; y++)
	{
		for(size_t x = 0; x < output_dims; x++)
		{
			output[y * output_dims + x] = input[y * input_dims + x];
		}
	}
}

void correlate_valid(const nnet_float *image, const size_t *image_dims, const nnet_float *kernel, const size_t *kernel_dims, nnet_float *output)
{
	/*size_t output_dims0 = image_dims[0] - kernel_dims[0] + 1;
	size_t output_dims1 = image_dims[1] - kernel_dims[1] + 1;

	for(size_t ky = 0; ky < kernel_dims[1]; ky++)
	{
		for(size_t kx = 0; kx < kernel_dims[0]; kx++)
		{
			VECTOR kvec = VSET1(*kernel);

			for(size_t y = 0; y < output_dims1; y++)
			{
				size_t x = 0;

				for(x = 0; x <= output_dims0 - VECTOR_LENGTH && output_dims0 >= VECTOR_LENGTH; x += VECTOR_LENGTH)
				{
					VSTORE(output + y * output_dims0 + x, VADD(VLOAD(output + y * output_dims0 + x), VMUL(kvec, VLOAD(image + (y + ky) * image_dims[0] + (x + kx)))));
				}
					
				for(; x < output_dims0; x++)
				{
					output[y * output_dims0 + x] += *kernel * image[(y + ky) * image_dims[0] + (x + kx)];
				}
			}

			kernel++;
		}
	}*/

	const size_t output_dims0 = image_dims[0] - kernel_dims[0] + 1;
	const size_t output_dims1 = image_dims[1] - kernel_dims[1] + 1;
	
	for(size_t y = 0; y < output_dims1; y++)
	{
		size_t x = 0;
		
		for(; x <= output_dims0 - VECTOR_LENGTH && output_dims0 >= VECTOR_LENGTH; x += VECTOR_LENGTH)
		{
			VECTOR accum = VLOAD(output);
			size_t offset = y * image_dims[0] + x;
			
			for(size_t ky = 0; ky < kernel_dims[1]; ky++)
			{
				for(size_t kx = 0; kx < kernel_dims[0]; kx++)
				{
					accum = VADD(accum, VMUL(VLOAD(image + ky * image_dims[0] + kx + offset), VSET1(kernel[ky * kernel_dims[0] + kx])));
				}
			}
			
			VSTORE(output, accum);
			output += VECTOR_LENGTH;
		}
		
		for(; x < output_dims0; x++)
		{
			nnet_float sum = *output;
			size_t offset = y * image_dims[0] + x;
			
			for(size_t ky = 0; ky < kernel_dims[1]; ky++)
			{
				for(size_t kx = 0; kx < kernel_dims[0]; kx++)
				{
					sum += image[ky * image_dims[0] + kx + offset] * kernel[ky * kernel_dims[0] + kx];
				}
			}
			
			*output = sum;
			output++;
		}
	}
}

void convolve_valid(const nnet_float *image, const size_t *image_dims, const nnet_float *kernel, const size_t *kernel_dims, nnet_float *output)
{
	/*size_t output_dims0 = image_dims[0] - kernel_dims[0] + 1;
	size_t output_dims1 = image_dims[1] - kernel_dims[1] + 1;
	kernel += kernel_dims[0] * kernel_dims[1];

	for(size_t ky = 0; ky < kernel_dims[1]; ky++)
	{
		for(size_t kx = 0; kx < kernel_dims[0]; kx++)
		{
			kernel--;
			VECTOR kvec = VSET1(*kernel);
			
			for(size_t y = 0; y < output_dims1; y++)
			{
				size_t x = 0;

				for(x = 0; x <= output_dims0 - VECTOR_LENGTH && output_dims0 >= VECTOR_LENGTH; x += VECTOR_LENGTH)
				{
					VSTORE(output + y * output_dims0 + x, VADD(VLOAD(output + y * output_dims0 + x), VMUL(kvec, VLOAD(image + (y + ky) * image_dims[0] + (x + kx)))));
				}

				for(; x < output_dims0; x++)
				{
					output[y * output_dims0 + x] += *kernel * image[(y + ky) * image_dims[0] + (x + kx)];
				}
			}
		}
	}*/

	const size_t output_dims0 = image_dims[0] - kernel_dims[0] + 1;
	const size_t output_dims1 = image_dims[1] - kernel_dims[1] + 1;
	
	for(size_t y = 0; y < output_dims1; y++)
	{
		size_t x = 0;
		
		for(; x <= output_dims0 - VECTOR_LENGTH && output_dims0 >= VECTOR_LENGTH; x += VECTOR_LENGTH)
		{
			VECTOR accum = VLOAD(output);
			size_t offset = y * image_dims[0] + x;
			
			for(size_t ky = 0; ky < kernel_dims[1]; ky++)
			{
				for(size_t kx = 0; kx < kernel_dims[0]; kx++)
				{
					accum = VADD(accum, VMUL(VLOAD(image + ky * image_dims[0] + kx + offset), VSET1(kernel[(kernel_dims[1] - ky - 1) * kernel_dims[0] + kernel_dims[0] - kx - 1])));
				}
			}
			
			VSTORE(output, accum);
			output += VECTOR_LENGTH;
		}
		
		for(; x < output_dims0; x++)
		{
			nnet_float sum = *output;
			size_t offset = y * image_dims[0] + x;
			
			for(size_t ky = 0; ky < kernel_dims[1]; ky++)
			{
				for(size_t kx = 0; kx < kernel_dims[0]; kx++)
				{
					sum += image[ky * image_dims[0] + kx + offset] * kernel[(kernel_dims[1] - ky - 1) * kernel_dims[0] + kernel_dims[0] - kx - 1];
				}
			}
			
			*output = sum;
			output++;
		}
	}
}

void correlate_full(const nnet_float *image, const size_t *image_dims, const nnet_float *kernel, const size_t *kernel_dims, nnet_float *output)
{
	size_t output_dims0 = image_dims[0] + kernel_dims[0] - 1;
	kernel += kernel_dims[0] * kernel_dims[1];	

	for(size_t ky = 0; ky < kernel_dims[1]; ky++)
	{
		for(size_t kx = 0; kx < kernel_dims[0]; kx++)
		{
			kernel--;
			VECTOR kvec = VSET1(*kernel);

			for(size_t y = 0; y < image_dims[1]; y++)
			{
				size_t x = 0;

				for(x = 0; x <= image_dims[0] - VECTOR_LENGTH && image_dims[0] >= VECTOR_LENGTH; x += VECTOR_LENGTH)
				{
					VSTORE(output + (y + ky) * output_dims0 + x + kx, VADD(VLOAD(output + (y +ky) * output_dims0 + x + kx), VMUL(kvec, VLOAD(image + y * image_dims[0] + x))));
				}

				for(; x < image_dims[0]; x++)
				{
					output[(y + ky) * output_dims0 + x + kx] += *kernel * image[y * image_dims[0] + x];
				}
			}
		}
	}	
}

void reverse(nnet_float *data, const size_t length)
{
	for(size_t i = 0; i < length / 2; i++)
	{
		nnet_float tmp = data[i];
		data[i] = data[length - i - 1];
		data[length - i - 1] = tmp;
	}
}

void rotate_180(nnet_float *image, size_t image_dims, nnet_float *output)
{
	output += image_dims * image_dims;

	for(size_t i = 0; i < image_dims * image_dims; i++)
	{
		output--;
		*output += *image;
		image++;
	}
}

void random_vector(nnet_float *vector, size_t length, nnet_float lower, nnet_float upper)
{
	for(size_t i = 0; i < length; i++)
	{
		vector[i] = (rand() / (nnet_float)RAND_MAX) * (upper - lower) + lower;
	}
}

void random_gaussian_vector(nnet_float *vector, size_t length, nnet_float mean, nnet_float stddev)
{
	default_random_engine rng;
	normal_distribution<nnet_float> N(mean, stddev);

	for(size_t i = 0; i < length; i++)
	{
		vector[i] = N(rng);
	}
}
