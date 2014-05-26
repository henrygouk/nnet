#include <cstring>
#include <ctime>
#include <iostream>
#include <sys/time.h>

#include <nnet/nnet.hpp>
#include "cifar10.hpp"
#include "evaluate.hpp"

using namespace std;

void createPatch(nnet_float *src, nnet_float *dst, size_t xOffset, size_t yOffset, bool flip)
{
	for(size_t c = 0; c < 3; c++)
	{
		for(size_t y = 0; y < 24; y++)
		{
			for(size_t x = 0; x < 24; x++)
			{
				if(!flip)
				{
					dst[y * 24 + x] = src[(y + yOffset) * 32 + x + xOffset];
				}
				else
				{
					dst[(23 - y) * 24 + 23 - x] = src[(y + yOffset) * 32 + x + xOffset];
				}
			}
		}

		src += 32 * 32;
		dst += 24 * 24;
	}
}

void createPatches(nnet_float *src, nnet_float *dst, size_t length)
{
	for(size_t i = 0; i < length; i++)
	{
		size_t yOffset = rand() % 9;
		size_t xOffset = rand() % 9;

		createPatch(src, dst, xOffset, yOffset, rand() % 2 == 0);

		src += 32 * 32 * 3;
		dst += 24 * 24 * 3;
	}
}

nnet_float evaluatePatches(X86FeedForward *ffnn, nnet_float *features, nnet_float *labels, size_t count)
{
	size_t xoffsets[] = {0, 8, 4, 0, 8};
	size_t yoffsets[] = {0, 0, 4, 8, 8};
	nnet_float *output = nnet_malloc(10);
	nnet_float *outputAccum = nnet_malloc(10);
	nnet_float *inputs = nnet_malloc(24 * 24 * 3);
	size_t correct = 0;

	for(size_t i = 0; i < count; i++)
	{
		memset(outputAccum, 0, sizeof(nnet_float) * 10);

		for(size_t j = 0; j < 1; j++)
		{
			createPatch(features + i * 32 * 32 * 3, inputs, 4, 4, false);//xoffsets[j], yoffsets[j]);
			ffnn->predict(inputs, output);
			
			for(size_t k = 0; k < 10; k++)
			{
				outputAccum[k] += output[k] / 5.0;
			}
		}

		size_t output_maxind = 0;
		nnet_float output_maxval = outputAccum[0];
		size_t labels_maxind = 0;
		nnet_float labels_maxval = labels[i * 10];

		for(size_t j = 1; j < 10; j++)
		{
			if(outputAccum[j] > output_maxval)
			{
				output_maxind = j;
				output_maxval = outputAccum[j];
			}

			if(labels[i * 10 + j] > labels_maxval)
			{
				labels_maxind = j;
				labels_maxval = labels[i * 10 + j];
			}
		}

		if(output_maxind == labels_maxind)
		{
			correct++;
		}
	}

	nnet_free(output);
	nnet_free(outputAccum);
	nnet_free(inputs);

	return (nnet_float)correct / (nnet_float)count;
}

int main(int argc, char **argv)
{
	X86ActivationFunction *rect = new X86RectifiedLinear();
	X86ActivationFunction *softmax = new X86Softmax();
	X86SGD *ur = new X86SGD();
	X86SGD *ur2 = new X86SGD();
	ur->learningRate = 0.001;
	ur->momentumRate = 0.9;
	ur2->learningRate = 0.001;
	ur2->momentumRate = 0.9;
	ur2->l2DecayRate = 0.004;

	size_t l1InputDims[] = {32, 32};
	size_t l2InputDims[] = {28, 28};
	size_t l3InputDims[] = {14, 14};
	size_t l4InputDims[] = {10, 10};
	size_t kernelDims[] = {5, 5};
	size_t poolDims[] = {2, 2};

	vector<X86Layer *> layers;
	layers.push_back(new X86Convolutional(2, l1InputDims, kernelDims, 3, 64, 0.0001, rect, ur));
	layers.push_back(new X86MaxPool(2, l2InputDims, 64, poolDims));
	layers.push_back(new X86Convolutional(2, l3InputDims, kernelDims, 64, 64, 0.01, rect, ur));
	layers.push_back(new X86MaxPool(2, l4InputDims, 64, poolDims));
	layers.push_back(new X86FullyConnected(64 * 5 * 5, 64, 0.01, rect, ur2));
	layers.push_back(new X86Dropout(64, 0.5));
	layers.push_back(new X86FullyConnected(64, 64, 0.01, rect, ur));
	layers.push_back(new X86Dropout(64, 0.5));
	layers.push_back(new X86FullyConnected(64, 10, 0.01, softmax, ur2));

	X86FeedForward *ff = new X86FeedForward(layers, new X86CrossEntropy());

	nnet_float *features, *labels;
	cifar10(argv[1], &features, &labels);

	nnet_float *patches = nnet_malloc(24 * 24 * 3 * 50000);

	cout << "Starting..." << endl;

	for(size_t i = 0; i < 120; i++)
	{
		nnet_shuffle_instances(features, labels, 40000, 32 * 32 * 3, 10);

		struct timeval start, end;
		gettimeofday(&start, 0);
		ff->train(features, labels, 40000, 1, 128);
		gettimeofday(&end, 0);

		cout << "Epoch: " << i + 1 << "\t";
		cout << "Training time: " << (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) << "\t";
		cout << "Training Accuracy: " << evaluate(ff, features, labels, 40000, 32 * 32 * 3, 10) << "\t";
		cout << "Validation Accuracy: " << evaluate(ff, features + 40000 * 32 * 32 * 3, labels + 40000 * 10, 10000, 32 * 32 * 3, 10) << endl;
	}

	/*for(size_t i = 0; i < 100; i++)
	{
		nnet_shuffle_instances(features, labels, 40000, 32 * 32 * 3, 10);

		createPatches(features, patches, 40000);
	
		struct timeval start, end;
		gettimeofday(&start, 0);
		ff->train(patches, labels, 40000, 1, 128);
		gettimeofday(&end, 0);

		cout << "Epoch: " << i + 1 << "\t";
		cout << "Training time: " << (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) << "\t";
		cout << "Training Accuracy: " << evaluatePatches(ff, features, labels, 40000) << "\t";
		cout << "Validation Accuracy: " << evaluatePatches(ff, features + 40000 * 32 * 32 * 3, labels + 40000 * 10, 10000) << endl;
	}

	for(size_t i = 0; i < 40; i++)
	{
		nnet_shuffle_instances(features, labels, 50000, 32 * 32 * 3, 10);

		createPatches(features, patches, 50000);
	
		struct timeval start, end;
		gettimeofday(&start, 0);
		ff->train(patches, labels, 50000, 1, 128);
		gettimeofday(&end, 0);

		cout << "Epoch: " << i + 1 << "\t";
		cout << "Training time: " << (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) << "\t";
		cout << "Training Accuracy: " << evaluatePatches(ff, features, labels, 40000) << "\t";
		cout << "Validation Accuracy: " << evaluatePatches(ff, features + 50000 * 32 * 32 * 3, labels + 50000 * 10, 10000) << endl;
	}

	ur->learningRate *= 0.1;
	ur2->learningRate *= 0.1;

	for(size_t i = 0; i < 10; i++)
	{
		nnet_shuffle_instances(features, labels, 50000, 32 * 32 * 3, 10);

		createPatches(features, patches, 50000);
	
		struct timeval start, end;
		gettimeofday(&start, 0);
		ff->train(patches, labels, 50000, 1, 128);
		gettimeofday(&end, 0);

		cout << "Epoch: " << i + 1 << "\t";
		cout << "Training time: " << (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) << "\t";
		cout << "Training Accuracy: " << evaluatePatches(ff, features, labels, 40000) << "\t";
		cout << "Validation Accuracy: " << evaluatePatches(ff, features + 50000 * 32 * 32 * 3, labels + 50000 * 10, 10000) << endl;
	}

	ur->learningRate *= 0.1;
	ur2->learningRate *= 0.1;

	for(size_t i = 0; i < 10; i++)
	{
		nnet_shuffle_instances(features, labels, 50000, 32 * 32 * 3, 10);

		createPatches(features, patches, 50000);
	
		struct timeval start, end;
		gettimeofday(&start, 0);
		ff->train(patches, labels, 50000, 1, 128);
		gettimeofday(&end, 0);

		cout << "Epoch: " << i + 1 << "\t";
		cout << "Training time: " << (double)(end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0) << "\t";
		cout << "Training Accuracy: " << evaluatePatches(ff, features, labels, 40000) << "\t";
		cout << "Validation Accuracy: " << evaluatePatches(ff, features + 50000 * 32 * 32 * 3, labels + 50000 * 10, 10000) << endl;
	}*/

	nnet_free(features);
	nnet_free(labels);
	nnet_free(patches);
}
