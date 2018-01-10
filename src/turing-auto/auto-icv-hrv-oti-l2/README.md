## Auto-Encoder : inv-hsv-osi-l2

This directory contains the implementation of a simple and single hidden layer
auto-encoder for raster images. It is trained using raster images data-sets in
greyscale format.

Its topology consist in a input layer that correspond in size to the total amount
of greyscale pixels of the data-set images. The input layer is connected to a
single hidden layer. The size of the unique hidden layer can be modulated using
the script parameters. The hidden layer is connected to the output layer that has
the same size as the input layer. The hidden layer has a sigmoid activation function
as the output layer.

The cost function of the network is a simple L2 norm between the input and output
layers. The theoretical minimum of the cost function is then the identity between
the input and output layer.

As the hidden layer can be modulated in size using the script parameters, it has
to be identical from a call to another on a specific data-set and trained
network. Loading a network that has been trained with a different hidden size
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
layer specific content to study the obtained output layer. Finally, the _view_
mode allows to dumps the parameters of the trained network to produce plots for
network deep analysis.

## Copyright and License

**turing-project** - Nils Hamel <br >
Copyright (c) 2016-2018 DHLAB, EPFL

The codes are licensed under the terms of the GNU GPLv3.
