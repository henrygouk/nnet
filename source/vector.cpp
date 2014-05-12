#include <cstdlib>
#include <cstring>

#include <nnet/vector.hpp>

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
#define VLOAD _mm256_load_ps
#define VLOADU _mm256_loadu_ps
#define VSTORE _mm256_store_ps
#define VSTOREU _mm256_storeu_ps

static inline nnet_float VHADD(VECTOR vec)
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
	__m256 b_flip = _mm256_shuffle_ps(b,b,0xB1);
	__m256 a_im = _mm256_shuffle_ps(a,a,0xF5);
	__m256 a_re = _mm256_shuffle_ps(a,a,0xA0);
	__m256 aib = _mm256_mul_ps(a_im, b_flip);
	__m256 arb = _mm256_mul_ps(a_re, b);
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

void vector_complex_fma(nnet_float *accum, nnet_float *a, nnet_float *b, size_t length)
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

void tensor_maxpool_nd(size_t rank, const nnet_float *input, const size_t *input_dims, const size_t *pool_dims, nnet_float *output, size_t *input_indices, size_t index)
{
	size_t input_volume = 1;
	size_t output_volume = 1;

	if(rank == 0)
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

void pad_rotate(size_t rank, const nnet_float *input, const size_t *input_dims, nnet_float *output, const size_t *output_dims)
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
		pad_rotate(rank - 1, input + (input_dims[0] - i - 1) * input_volume, input_dims + 1, output + i * output_volume, output_dims + 1);
	}

	memset(output + i * output_volume, 0, sizeof(nnet_float) * output_volume * (output_dims[0] - input_dims[0]));
}

void extract_valid(size_t rank, const nnet_float *input, const size_t *input_dims, nnet_float *output, const size_t *output_dims)
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

	size_t offset = input_dims[0] - output_dims[0];

	for(i = 0; i < output_dims[0]; i++)
	{
		extract_valid(rank - 1, input + (i + offset) * input_volume, input_dims + 1, output + i * output_volume, output_dims + 1);
	}
}

void extract_valid_rotate(size_t rank, const nnet_float *input, const size_t *input_dims, nnet_float *output, const size_t *output_dims)
{
	size_t input_volume = 1;
	size_t output_volume = 1;
	size_t i;

	if(rank == 0)
	{
		*output += *input;
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
		extract_valid_rotate(rank - 1, input + (i + offset) * input_volume, input_dims + 1, output + (output_dims[0] - i - 1) * output_volume, output_dims + 1);
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

void correlate_valid(nnet_float *image, size_t image_dims, nnet_float *kernel, size_t kernel_dims, nnet_float *output)
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

void convolve_valid(nnet_float *image, size_t image_dims, nnet_float *kernel, size_t kernel_dims, nnet_float *output)
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

void correlate_full(nnet_float *image, size_t image_dims, nnet_float *kernel, size_t kernel_dims, nnet_float *output)
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
