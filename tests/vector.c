#include <stdio.h>
#include <string.h>

#include "../source/core.h"
#include "../source/vector.h"

void test_valid_convolve(void)
{
	int passed = 0;

	printf("test_valid_convolve: ");
	fflush(stdout);

	//Generate some test stimulus
	nnet_float_t *input = nnet_malloc(9);
	nnet_float_t *kernel = nnet_malloc(9);
	nnet_float_t *output = nnet_malloc(4);

	memset(output, 0, sizeof(nnet_float_t) * 4);

	for(size_t i = 0; i < 9; i++)
	{
		input[i] = i + 1;
		kernel[i] = i + 1;
	}

	convolve_valid(input, 3, kernel, 3, output);

	if(output[0] == 165.0)
		passed++;

	output[0] = 0.0;

	convolve_valid(input, 3, kernel, 2, output);

	if(output[0] == 23 && output[1] == 33 && output[2] == 53 && output[3] == 63)
		passed++;
	
	printf("%d/2 passed\n", passed);

	nnet_free(input);
	nnet_free(kernel);
	nnet_free(output);
}

void test_valid_correlation(void)
{
	int passed = 0;

	printf("test_valid_correlate: ");
	fflush(stdout);

	//Generate some test stimulus
	nnet_float_t *input = nnet_malloc(9);
	nnet_float_t *kernel = nnet_malloc(9);
	nnet_float_t *output = nnet_malloc(4);

	memset(output, 0, sizeof(nnet_float_t) * 4);

	for(size_t i = 0; i < 9; i++)
	{
		input[i] = i + 1;
		kernel[i] = i + 1;
	}

	correlate_valid(input, 3, kernel, 3, output);

	if(output[0] == 285)
		passed++;

	output[0] = 0.0;

	correlate_valid(input, 3, kernel, 2, output);

	if(output[0] == 37 && output[1] == 47 && output[2] == 67 && output[3] == 77)
		passed++;

	printf("%d/2 passed\n", passed);

	nnet_free(input);
	nnet_free(kernel);
	nnet_free(output);
}

void test_full_correlation(void)
{
	int passed = 0;

	printf("test_full_correlate: ");
	fflush(stdout);

	nnet_float_t *input = nnet_malloc(9);
	nnet_float_t *kernel = nnet_malloc(9);
	nnet_float_t *output = nnet_malloc(25);
	nnet_float_t correct[25] = {9, 26, 50, 38, 21, 42, 94, 154, 106, 54, 90, 186, 285, 186, 90, 54, 106, 154, 94, 42, 21, 38, 50, 26, 9};
	
	memset(output, 0, sizeof(nnet_float_t) * 25);

	for(size_t i = 0; i < 9; i++)
	{
		input[i] = i + 1;
		kernel[i] = i + 1;
	}

	correlate_full(input, 3, kernel, 3, output);

	int c = 0;

	for(size_t i = 0; i < 25; i++)
	{
		if(output[i] == correct[i])
			c++;
	}

	if(c == 25)
		passed++;

	printf("%d/1 passed\n", passed);

	nnet_free(input);
	nnet_free(kernel);
	nnet_free(output);
}

int main(int argc, char **argv)
{
	test_valid_convolve();
	test_valid_correlation();
	test_full_correlation();

	return 0;
}
