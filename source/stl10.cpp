#include <fstream>

#include "nnet/stl10.hpp"

using namespace std;

void loadImages(const char *filename, vector<nnet_float> &features)
{
	size_t bufSize = 96 * 96 * 3;
	unsigned char *buf = new unsigned char[bufSize];

	ifstream inputFile(filename, ios::binary);

	while(true)
	{
		inputFile.read((char *)buf, bufSize);

		if(inputFile.eof())
		{
			break;
		}

		//Convert each channel into row major order
		for(size_t c = 0; c < 3; c++)
		{
			for(size_t y = 0; y < 96; y++)
			{
				for(size_t x = 0; x < 96; x++)
				{
					features.push_back(buf[c * 96 * 96 + x * 96 + y]);
				}
			}
		}
	}

	inputFile.close();

	delete[] buf;
}

void loadLabels(const char *filename, vector<nnet_float> &labels)
{
	char c;

	ifstream inputFile(filename, ios::binary);

	while(true)
	{
		inputFile.get(c);

		if(inputFile.eof())
		{
			break;
		}

		//Labels are stored in the range 1-10
		c--;

		for(int i = 0; i < 10; i++)
		{
			if(c == i)
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

void loadIndices(const char *filename, vector<size_t> &indices)
{
	size_t ind;

	ifstream inputFile(filename, ios::binary);

	while(true)
	{
		inputFile >> ind;

		if(inputFile.eof())
		{
			break;
		}

		indices.push_back(ind);
	}

	inputFile.close();
}

size_t loadStl10(const vector<const char *> &filenames, vector<nnet_float> &features, vector<nnet_float> &labels, vector<nnet_float> &unlabelled, vector<size_t> &indices)
{
	//Load the labelled images into the features vector
	loadImages(filenames[0], features);

	//Load the labels
	loadLabels(filenames[1], labels);

	//Load the unlabelled images
	loadImages(filenames[2], unlabelled);

	//Load the fold indices
	loadIndices(filenames[3], indices);

	return labels.size() / 10;
}
