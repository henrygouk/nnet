nnet
====

High performance Artificial Neural Network library.

Dependencies:
* FFTW3

As you may have guessed from the dependency on FFTW, this library uses FFT convolutions by default.

Building
--------

We use the CMake build system. Here is an example of how you might build the library:

	mkdir build && cd build
	cmake ..
	make
	
Then to install, run this as root:

	make install

If your CPU has access to AVX instructions you should use the `-DENABLE_AVX=ON` flag when running cmake.

Similarly, if your CPU has access to FMA instructions the `-DENABLE_FMA=ON` flag should be used as well.

Example
-------

Included is an example convolutional network for classifying MNIST. The network is based on one described in [1]. The program expects five filenames to be passed as command line arguments:

`mnistcnn <train-images> <test-images> <train-labels> <test-labels> <output_model>`

The first four files are the input MNIST data files from Yann LeCun's website [2]. The last filename is where the trained model is stored.

[1] Scherer, D., Müller, A., & Behnke, S. (2010). Evaluation of pooling operations in convolutional architectures for object recognition. In Artificial Neural Networks–ICANN 2010 (pp. 92-101). Springer Berlin Heidelberg.

[2] http://yann.lecun.com/exdb/mnist/

Reference
---------

Please cite the following reference in papers using this library:

Gouk, H. G., & Blake, A. M. (2014, November). Fast Sliding Window Classification with Convolutional Neural Networks. In Proceedings of the 29th International Conference on Image and Vision Computing New Zealand (p. 114). ACM.
