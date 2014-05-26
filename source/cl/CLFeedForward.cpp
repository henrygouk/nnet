#include "nnet/cl/CLFeedForward.hpp"

using namespace std;

CLFeedForward::CLFeedForward(const vector<CLLayer *> &layervec, CLLoss *lf)
{
	vector<cl::Platform> platforms;

	cl::Platform::get(&platforms);

	if(platforms.size() == 0)
	{
		//TODO: there is a problem
	}

	cl::Platform platform = platforms[0];

	vector<cl::Device> devices;

	if(devices.size() == 0)
	{
		//TODO: there is a problem
	}

	cl::Device device = devices[0];

	context = new cl::Context({device});
}

CLFeedForward::~CLFeedForward()
{
	delete context;
}

void CLFeedForward::train(const nnet_float *features, const nnet_float *labels, const std::size_t numInstances, int epochs, int batchSize)
{
	//Load the instances into the GPU
	cl::Buffer featuresBuffer(context, CL_MEM_READ, sizeof(nnet_float) * numFeatures * numInstances);
	cl::Buffer labelsBuffer(context, CL_MEM_READ, sizeof(nnet_float) * numLabels * numInstances);
	queue.enqueueWriteBuffer(featuresBuffer, CL_TRUE, 0, sizeof(nnet_float) * numFeatures * numInstances);
	queue.enqueueWriteBuffer(labelsBuffer, CL_TRUE, 0, sizeof(nnet_float) * numLabels * numInstances);

	for(int e = 0; e < epochs; e++)
	{
		for(int i = 0; i < numInstances; i++)
		{
			for_each(layers.begin(), layers.end(), [] (CLLayer *l) { l->startBatch(); });

			for(size_t j = 0; j < batchSize; j++)
			{
				
			}

			for_each(layers.begin(), layers.end(), [] (CLLayer *l) { l->endBatch(); });
		}
	}
}
