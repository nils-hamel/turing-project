## Auto-Encoder - Image - inv-c5rv-d5si-l2

This directory contains the implementation of a simple convolutional auto-encoder
for image raster. It is trained using raster images data-sets in
greyscale format.

The topology of the network consist in the first place in a convolutional layer
activated by a _relu_ function. Directly after this first layer, a second
convolutional layer is set to rebuild the original raster with a sigmoid as
activation layer.

The cost function of the network is a simple L2 norm between the input and output
layers. The theoretical minimum of the cost function is then the identity between
the input and output layer.

As the convolution depth can be modulated in size, using the script parameters,
it has to be identical from a call to another on a specific data-set and trained
network. Loading a network that has been trained with a different size
will raise an error.

The input layer is fed using the raster image from the specified data-set. The
image are initially converted into greyscale image before to be normalised on the
[0,1] floating point range. The output image have then to be expected in the same
format.

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
