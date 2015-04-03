#include <fstream>
#include <iostream>

#include "nnet/cifar10.hpp"

using namespace std;

size_t loadCifar10(const vector<string> &filenames, vector<nnet_float> &features, vector<nnet_float> &labels)
{
	size_t bufSize = 32 * 32 * 3 + 1;
	unsigned char *buf = new unsigned char[bufSize];

	for(size_t i = 0; i < filenames.size(); i++)
	{
		ifstream inputFile(filenames[i].c_str(), ios::binary);

		while(true)
		{
			inputFile.read((char *)buf, bufSize);

			if(inputFile.eof())
			{
				break;
			}

			for(size_t j = 1; j < bufSize; j++)
			{
				features.push_back((nnet_float)buf[j] / 255.0);
			}

			for(size_t j = 0; j < 10; j++)
			{
				if(buf[0] == j)
				{
					labels.push_back(1.0);
				}
				else
				{
					labels.push_back(0.0);
				}
			}
		}

		inputFile.close();
	}

	delete[] buf;

	return labels.size() / 10;
}
