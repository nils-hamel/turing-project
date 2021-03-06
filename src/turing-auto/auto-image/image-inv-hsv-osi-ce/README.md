## Auto-Encoder - Image - inv-hsv-osi-ce

This directory contains the implementation of a simple and single hidden layer
auto-encoder for raster images. It is trained using raster images data-sets in
greyscale format.

Its topology consist in a input layer that correspond in size to the total amount
of greyscale pixels of the data-set images. The input layer is connected to a
single hidden layer. The size of the unique hidden layer can be modulated using
the script parameters. The hidden layer is connected to the output layer that has
the same size as the input layer. The hidden layer has a sigmoid activation function
as the output layer.

The network uses cross-entropy between the input and output layer as cost function.

As the hidden layer can be modulated in size using the script parameters, it has
to be identical from a call to another on a specific data-set and trained
network. Loading a network that has been trained with a different hidden size
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
