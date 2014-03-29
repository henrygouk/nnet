#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../source/core.h"
#include "../source/types.h"

/*
    Loads a file containing a dataset of images
*/
nnet_float_t *cifar10_images(const char *filename)
{
    //Open the file
    int fp = open(filename, O_RDWR);
  
    //Find out how long it is  
    lseek(fp, 0, SEEK_END);
    size_t file_length = lseek(fp, 0, SEEK_CUR);
    lseek(fp, 0, SEEK_SET);
    
    //Memory map it
    uint8_t *file = mmap(0, file_length, PROT_READ, MAP_SHARED, fp, 0);
    
    //Get the dimensions of the images, the number of channels, and the number of instances
    uint32_t dims = ((uint32_t *)file)[0];
    uint32_t chans = ((uint32_t *)file)[1];
    uint32_t num_images = ((uint32_t *)file)[2];
    file += 3 * sizeof(uint32_t);
    
    //Calculate the image size and the padded image size (for cache friendly alignment)
    size_t img_size = dims * dims;
    
    //Allocate memory to store the instances
    nnet_float_t *instances = nnet_malloc(img_size * chans * num_images * sizeof(nnet_float_t));
    
    //Start reading the instances in
    for(size_t i = 0; i < num_images; i++)
    {
        for(size_t j = 0; j < chans; j++)
        {
            for(size_t k = 0; k < img_size; k++)
            {
                instances[(i * chans + j) * img_size + k] = (nnet_float_t)file[(i * chans + j) * img_size + k] / 255.0;
            }
        }
    }
    
    //Close the file
    munmap(file - 3 * sizeof(uint32_t), file_length);
    close(fp);
    
    return instances;
}

/*
	Loads the labels for a dataset
*/
nnet_float_t *cifar10_labels(const char *filename, nnet_float_t pos_value, nnet_float_t neg_value)
{
    //Open the file
    int fp = open(filename, O_RDWR);
  
    //Find out how long it is  
    lseek(fp, 0, SEEK_END);
    size_t file_length = lseek(fp, 0, SEEK_CUR);
    lseek(fp, 0, SEEK_SET);
    
    //Memory map it
    uint8_t *file = mmap(0, file_length, PROT_READ, MAP_SHARED, fp, 0);
    
    //Get the number of labels and number of instances
    uint32_t num_labels = ((uint32_t *)file)[0];
    uint32_t num_insts = ((uint32_t *)file)[1];
    file += 2 * sizeof(uint32_t);
    
    //Allocate memory to store the labels
    nnet_float_t *labels = nnet_malloc(num_labels * num_insts * sizeof(nnet_float_t));
    
    //Start reading the instances in
    for(size_t i = 0; i < num_insts; i++)
    {
		for(size_t k = 0; k < num_labels; k++)
		{
			if(file[i] == k)
			{
				labels[i * num_labels + k] = pos_value;
			}
			else
			{
				labels[i * num_labels + k] = neg_value;
			}
		}
    }
    
    //Close the file
    munmap(file - 2 * sizeof(uint32_t), file_length);
    close(fp);
    
    return labels;
}
