## Auto-Encoder : icv-htv-oti-l2

This directory contains the implementation of a single layer auto-encoder for
raster images. It is trained using raster images data-sets in greyscale format.

The topology of the network starts with the input layer with a size that matches
the amount of greyscale pixels of the input images. The input layer is connected
to the unique hidden layer that has a size that can be modulated using the script
parameters. The hidden layer is then directly connected to the output layer that
has the same size as the input layer. Both hidden and output layers are activated
through the hyperbolic tangent function.

The cost function of the network is a simple L2 norm between the input and output
layers. Theoretically, the strict minimum of the cost function correspond to the
perfect identity of the two input and output layers.

As the hidden layer can be modulated in size using the script parameters, its
size has to be identical from a call to another. Loading a trained network with
a given hidden size with another value as parameters will raise an error.

As hyperbolic tangent is used as activation function for both hidden and output
layers imply specificity in the image input and output format. Instead of being
normalised on the [0,1] floating point range, the greyscale pixels intensities
are normalised on [-1,+1] range. The image produced by the output layer are then
expressed according to this specific format.

The network implementation script provides different modes. The first mode
correspond to the network training and retraining process. The second mode
allows to auto-encode image of the data-set by specified the image index range.
The two next mode allows to encode and decode specific image of the data-set again
using index range. The encoding produce a file in which the state of the central hidden
layer is dumped, in text mode. The decoding mode allows to provides to the central hidden
layer specific content to study the obtained output layer.

## Copyright and License

**turing-project** - Nils Hamel <br >
Copyright (c) 2016-2018 DHLAB, EPFL

The codes are licensed under the terms of the GNU GPLv3.