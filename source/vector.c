#include <string.h>

#include "vector.h"

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
#define VLOAD _mm256_load_ps
#define VLOADU _mm256_loadu_ps
#define VSTORE _mm256_store_ps
#define VSTOREU _mm256_storeu_ps

static inline nnet_float_t VHADD(VECTOR vec)
{
	VECTOR vec2 = _mm256_permute2f128_ps(vec, vec, 1);
	vec = _mm256_add_ps(vec, vec2);
	vec = _mm256_hadd_ps(vec, vec);
	vec = _mm256_hadd_ps(vec, vec);
	return _mm_cvtss_f32(_mm256_castps256_ps128(vec));
}

static VECTOR conj_vector;

static inline VECTOR VCMUL(VECTOR a, VECTOR b)
{
	__m256 re, im;
	re = _mm256_shuffle_ps(a,a,_MM_SHUFFLE(2,2,0,0));
	re = _mm256_mul_ps(re, b);

	im = _mm256_shuffle_ps(a,a,_MM_SHUFFLE(3,3,1,1));
	b = _mm256_shuffle_ps(b,b,_MM_SHUFFLE(2,3,0,1));
	im = _mm256_mul_ps(im, b);
	im = _mm256_xor_ps(im, conj_vector);
	
	return _mm256_add_ps(re, im);
}

#else

//Fall back to scalar arithmetic

#define VECTOR nnet_float_t
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

nnet_float_t dot_product(nnet_float_t *vec1, nnet_float_t *vec2, size_t length)
{
	nnet_float_t accum = 0;
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

nnet_float_t vector_sum(nnet_float_t *vec, size_t length)
{
	size_t i = 0;
	nnet_float_t res = 0;
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

void matrix_vector_mul(nnet_float_t *matrix, size_t rows, size_t cols, nnet_float_t *vecin, nnet_float_t *vecout)
{
	for(size_t r = 0; r < rows; r++)
	{
		vecout[r] = dot_product(vecin, matrix + r * cols, cols);
	}
}

void matrix_trans_vector_mul(nnet_float_t *matrix, size_t rows, size_t cols, nnet_float_t *vecin, nnet_float_t *vecout)
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

void vector_accum(nnet_float_t *vec1, nnet_float_t *vec2, size_t length)
{
	size_t i = 0;

	for(; i < length; i++)
	{
		vec1[i] += vec2[i];
	}
}

void vector_mul(nnet_float_t *vec1, nnet_float_t *vec2, nnet_float_t *output, size_t length)
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

void vector_scale_accum(nnet_float_t *vec1, nnet_float_t *vec2, nnet_float_t scalar, size_t length)
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

void vector_complex_fma(nnet_float_t *accum, nnet_float_t *a, nnet_float_t *b, size_t length)
{
	size_t i = 0;

#if HAVE_AVX

	conj_vector = VSET(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);

	for(; i <= length - (VECTOR_LENGTH >> 1) && length >= (VECTOR_LENGTH >> 1); i += (VECTOR_LENGTH >> 1))
	{
		VSTORE(accum, VADD(VLOAD(accum), VCMUL(VLOAD(a), VLOAD(b))));
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

void vector_complex_conj_fma(nnet_float_t *accum, nnet_float_t *a, nnet_float_t *b, size_t length)
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

void vector_scale(nnet_float_t *vector, size_t length, nnet_float_t scalar)
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

void pad(nnet_float_t *input, size_t input_dims, nnet_float_t *output, size_t output_dims)
{
	memset(output, 0, sizeof(nnet_float_t) * output_dims * output_dims);

	for(size_t y = 0; y < input_dims; y++)
	{
		for(size_t x = 0; x < input_dims; x++)
		{
			output[y * output_dims + x] = input[y * input_dims + x];
		}
	}
}

void pad_rotate(nnet_float_t *input, size_t input_dims, nnet_float_t *output, size_t output_dims)
{
	memset(output, 0, sizeof(nnet_float_t) * output_dims * output_dims);

	for(size_t y = 0; y < input_dims; y++)
	{
		for(size_t x = 0; x < input_dims; x++)
		{
			output[(input_dims - y - 1) * output_dims + input_dims - x - 1] = input[y * input_dims + x];
		}
	}
}

void extract_valid(nnet_float_t *input, size_t input_dims, nnet_float_t *output, size_t output_dims, size_t kernel_dims)
{
	size_t offset = kernel_dims - 1;

	for(size_t y = 0; y < output_dims; y++)
	{
		for(size_t x = 0; x < output_dims; x++)
		{
			output[y * output_dims + x] = input[(y + offset) * input_dims + (x + offset)];
		}
	}
}

void extract_valid_rotate(nnet_float_t *input, size_t input_dims, nnet_float_t *output, size_t output_dims, size_t kernel_dims)
{
	size_t offset = kernel_dims - 1;

	for(size_t y = 0; y < output_dims; y++)
	{
		for(size_t x = 0; x < output_dims; x++)
		{
			output[(output_dims - y - 1) * output_dims + output_dims - x - 1] = input[(y + offset) * input_dims + (x + offset)];
		}
	}
}

void extract_full_rotate(nnet_float_t *input, size_t input_dims, nnet_float_t *output, size_t output_dims)
{
	for(size_t y = 0; y < output_dims; y++)
	{
		for(size_t x = 0; x < output_dims; x++)
		{
			output[(output_dims - y - 1) * output_dims + (output_dims - x - 1)] = input[y * input_dims + x];
		}
	}
}

void extract_full(nnet_float_t *input, size_t input_dims, nnet_float_t *output, size_t output_dims)
{
	for(size_t y = 0; y < output_dims; y++)
	{
		for(size_t x = 0; x < output_dims; x++)
		{
			output[y * output_dims + x] = input[y * input_dims + x];
		}
	}
}

void correlate_valid(nnet_float_t *image, size_t image_dims, nnet_float_t *kernel, size_t kernel_dims, nnet_float_t *output)
{
	size_t output_dims = image_dims - kernel_dims + 1;

	for(size_t ky = 0; ky < kernel_dims; ky++)
	{
		for(size_t kx = 0; kx < kernel_dims; kx++)
		{
			for(size_t y = 0; y < output_dims; y++)
			{
				for(size_t x = 0; x < output_dims; x++)
				{
					output[y * output_dims + x] += *kernel * image[(y + ky) * image_dims + (x + kx)];
				}
			}

			kernel++;
		}
	}
}

void convolve_valid(nnet_float_t *image, size_t image_dims, nnet_float_t *kernel, size_t kernel_dims, nnet_float_t *output)
{
	size_t output_dims = image_dims - kernel_dims + 1;
	kernel += kernel_dims * kernel_dims;

	for(size_t ky = 0; ky < kernel_dims; ky++)
	{
		for(size_t kx = 0; kx < kernel_dims; kx++)
		{
			kernel--;

			for(size_t y = 0; y < output_dims; y++)
			{
				for(size_t x = 0; x < output_dims; x++)
				{
					output[y * output_dims + x] += *kernel * image[(y + ky) * image_dims + (x + kx)];
				}
			}
		}
	}
}

void correlate_full(nnet_float_t *image, size_t image_dims, nnet_float_t *kernel, size_t kernel_dims, nnet_float_t *output)
{
	size_t output_dims = image_dims + kernel_dims - 1;
	kernel += kernel_dims * kernel_dims;

	for(size_t ky = 0; ky < kernel_dims; ky++)
	{
		for(size_t kx = 0; kx < kernel_dims; kx++)
		{
			kernel--;

			for(size_t y = 0; y < image_dims; y++)
			{
				for(size_t x = 0; x < image_dims; x++)
				{
					output[(y + ky) * output_dims + x + kx] += *kernel * image[y * image_dims + x];
				}
			}
		}
	}
}

void rotate_180(nnet_float_t *image, size_t image_dims, nnet_float_t *output)
{
	output += image_dims * image_dims;

	for(size_t i = 0; i < image_dims * image_dims; i++)
	{
		output--;
		*output += *image;
		image++;
	}
}
