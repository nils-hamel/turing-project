## Auto-Encoder : inv-c5rv-mp-up-d5si-l2

This directory contains the implementation of a simple convolutional auto-encoder
for image raster. It is trained using raster images data-sets in
greyscale format.

The topology of the network consist in the first place in a convolutional layer
followed by a max-pool layer. The activation function of the convolutional layer
is a _relu_ function. Following the max-pool layer, an upscale operation is set
to retrieve the raster original resolution. A deconvolution layer is finally
used to reduce the convolution depth to one. In addition, the activation function
of the deconvolution layer is a sigmoid function.

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
correspond to the network training and retraining process. The second mode
allows to auto-encode image of the data-set by specified the image index range.
The two next mode allows to encode and decode specific image of the data-set again
using index range. The encoding produce a file in which the state of the hidden
layer is dumped, in text mode. The decoding mode allows to provides to the hidden
layer specific content to study the obtained output layer.

## Copyright and License

**turing-project** - Nils Hamel <br >
Copyright (c) 2016-2018 DHLAB, EPFL

The codes are licensed under the terms of the GNU GPLv3.
