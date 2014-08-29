#include <fstream>
#include <iostream>
#include <cstdio>
#include <cfloat>

using namespace std;

float euclideanDistance(float *a, float *b)
{
	float accum = 0.0;

	for(size_t i = 0; i < 64; i++)
	{
		accum += (a[i] - b[i]) * (a[i] - b[i]);
	}

	return accum;
}

int main(int argc, char **argv)
{
	FILE *fd = fopen(argv[1], "rb");
	float *buf = new float[64 * 10000];
	cout << fread(buf, sizeof(float), 64 * 10000, fd) << endl;
	fclose(fd);

	size_t qind = 7409;
	size_t rind = qind;
	float minDist = FLT_MAX;

	for(size_t i = 0; i < 10000; i++)
	{
		if(i == qind)
			continue;

		float d = euclideanDistance(buf + i * 64, buf + qind * 64);

		if(d < minDist)
		{
			rind = i;
			minDist = d;
		}
	}

	cout << minDist << " " << rind << endl;

	return 0;
}
