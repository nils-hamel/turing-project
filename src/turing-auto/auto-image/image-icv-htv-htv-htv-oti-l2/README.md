## Auto-Encoder : icv-htv-htv-htv-oti-l2

This directory contains the implementation of a three hidden layer auto-encoder
for raster images. It is trained using raster images data-sets in greyscale format.

The topology of the network starts with the input layer that has its size identical
to the number of pixel of the data-set images. The input layer is connected to
a first hidden layer with a size that can be modulated through the script parameters.
This hidden layer is connected to the central hidden layer that can also be modulated
in size. The central hidden layer is connected to the third hidden layer. The
size of the third hidden layer is always identical to the size of the first hidden
layer. The third hidden layer is finally connected to the output layer that has
the size of the input layer. All hidden layers and the output layer comes with a
hyperbolic tangent as activation function.

The cost function of the network is a simple L2 norm between the input and output
layers. The theoretical minimum of the cost function is then the identity between
the input and output layer.

As the hiddens layers can be modulated in size using the script parameters, they have
to be identical from a call to another on a specific data-set and trained
network. Loading a network that has been trained with different hidden sizes
will raise an error.

Using hyperbolic tangent as activation function implies specificity in the
image input and output format. Instead of being normalised on the [0,1] floating
point range, the greyscale pixels intensities are normalised on [-1,+1] range.
The images created by the output layer are also normalised on this specific
range.

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