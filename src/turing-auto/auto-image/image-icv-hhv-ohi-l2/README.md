## Auto-Encoder - Image - icv-hhv-ohi-l2

This directory contains the implementation of a single layer auto-encoder for
raster images. It is trained using raster images data-sets in greyscale format.

The topology of the network starts with the input layer with a size that matches
the amount of greyscale pixels of the input images. The input layer is connected
to the unique hidden layer that has a size that can be modulated using the script
parameters. The hidden layer is then directly connected to the output layer that
has the same size as the input layer. The hidden and output layer have both
invert hyperbolic sinus as activation function.

The cost function of the network is a simple L2 norm between the input and output
layers. Theoretically, the strict minimum of the cost function correspond to the
perfect identity of the two input and output layers.

As the hidden layer can be modulated in size using the script parameters, its
size has to be identical from a call to another. Loading a trained network with
a given hidden size with another value as parameters will raise an error.

Using invert hyperbolic sinus as activation function implies specificity in the
image input and output format. Instead of being normalised on the [0,1] floating
point range, the greyscale pixels intensities are normalised on [-1,+1] range.
The images created by the output layer are also normalised on this specific
range.

The network implementation script provides different modes. The first mode
corresponds to the network training and retraining process. The second mode
allows to auto-encode images of the data-set by specifying the range of the
images to consider. The next mode allows to encode a specified range of images
of the data-set in the auto-encoder latent space. The output of this mode is
then a set of vectors. The last mode allows to read a set of vectors interpreted
as the auto-encoder latent values to produce to corresponding images.

## Copyright and License

**turing-project** - Nils Hamel <br >
Copyright (c) 2016-2018 DHLAB, EPFL

The codes are licensed under the terms of the GNU GPLv3.
