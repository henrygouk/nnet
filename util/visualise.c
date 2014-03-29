#include <float.h>

#include <SDL/SDL.h>

#include "../source/fftconv.h"
#include "../source/types.h"

SDL_Surface *screen;

void visual_init(void)
{
	SDL_Init(SDL_INIT_EVERYTHING);

	screen = SDL_SetVideoMode(800, 600, 32, SDL_SWSURFACE);
}

void visual_layer(layer_t *layer, int y)
{
	fftconv_layer_data_t *layer_data = layer->layer_data;
	SDL_Surface *kernel = SDL_CreateRGBSurface(SDL_SWSURFACE, layer_data->kernel_dims * 2, layer_data->kernel_dims * 2, 32, 0, 0, 0, 0);

	nnet_float_t *weights = layer->weights;

	for(size_t i = 0; i < layer_data->num_input_maps * layer_data->num_output_maps; i++)
	{
		nnet_float_t minVal = FLT_MAX;
		nnet_float_t maxVal = -FLT_MAX;

		for(size_t j = 0; j < layer_data->kernel_dims * layer_data->kernel_dims; j++)
		{
			if(weights[j] < minVal)
				minVal = weights[j];

			if(weights[j] > maxVal)
				maxVal = weights[j];
		}

		for(size_t y = 0; y < layer_data->kernel_dims; y++)
		{
			for(size_t x = 0; x < layer_data->kernel_dims; x++)
			{
				size_t j = 4 * y * layer_data->kernel_dims + x * 2;
				unsigned int val = 255.0 * ((weights[j] - minVal) / (maxVal - minVal));
				val = val > 255 ? 255 : val;
				int word = (255 << 24) | (val << 16) | (val << 8) | val;
				((int *)kernel->pixels)[j] = word;
				((int *)kernel->pixels)[j + 1] = word;
				j += layer_data->kernel_dims * 2;
				((int *)kernel->pixels)[j] = word;
				((int *)kernel->pixels)[j + 1] = word;
			}
		}

		SDL_Rect rect;
		rect.x = i * (layer_data->kernel_dims + 2) * 2;
		rect.y = 2 + y;
		rect.w = 10;
		rect.h = 10;

		SDL_BlitSurface(kernel, 0, screen, &rect);

		weights += layer_data->kernel_dims * layer_data->kernel_dims;
	}

	SDL_Flip(screen);

	SDL_FreeSurface(kernel);
}

void visual_dispose(void)
{
	SDL_Quit();
}
