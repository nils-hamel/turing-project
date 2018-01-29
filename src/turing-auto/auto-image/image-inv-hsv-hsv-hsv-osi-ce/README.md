## Auto-Encoder - Image - inv-hsv-hsv-hsv-osi-ce

This directory contains the implementation of a three hidden layer auto-encoder
for raster images. It is trained using raster images data-sets in grayscale format.

The topology of the network starts with the input layer that has its size identical
to the number of pixel of the data-set images. The input layer is connected to
a first hidden layer with a size that can be modulated through the script parameter.
This hidden layer is connected to the central hidden layer that can also be modulated
in size. The central hidden layer is connected to the third hidden layer. The
size of the third hidden layer is always identical to the size of the first hidden
layer. The third hidden layer is finally connected to the output layer that has
the size of the input layer. All hidden layer and the output layer comes with a
sigmoid as activation function.

The network uses cross-entropy between the input and output layer as cost function.

As the hidden layers can be modulated in size using the script parameters, they have
to be identical from a call to another on a specific data-set and trained
network. Loading a network that has been trained with different hidden sizes
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
