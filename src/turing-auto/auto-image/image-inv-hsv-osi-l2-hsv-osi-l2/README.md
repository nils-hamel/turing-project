## Auto-Encoder - Image - inv-hsv-osi-l2-hsv-osi-l2

This directory contains the implementation of a double-layer and double-loss
auto-encoder for raster images. It is trained using raster images data-set in
greyscale format.

The topology of the auto-encoder is split in two sub-topology allowing the
definition of two loss functions during training. The first sub-topology is
a super-scaler network. Its input layer expects image with width divided by
two. An hidden layer with adjustable size is connected to the input layer. The
output layer is directly connected to the hidden layer and produce the super-scaled
version of the image. The hidden and output layers come with a sigmoid as
activation function.

The second sub-topology is the auto-encoder itself. The input layer expects the
images in grayscale format and is connected to a single hidden layer with adjustable
size. The hidden layer is then connected to the output layer that produces images
with a width divided by two according to the input layer. Both hidden and output
layer come with a sigmoid activation function.

The overall auto-encoder consists in the sub-auto-encoder followed by the super-scaler
in order to increase the resolution of the images coming out of the sub-auto-encoder.
This way of defining the network allows specific training. As the sub-auto-encoder
is trained with image and their down-sampled version, the super-scaler is trained
reversely. This allows to train both sub-network independently focusing on their 
specific task : auto-encoding and super-scaling. This follows in some way th
idea implemented in [1]

Both cost function of the sub-networks are simple L2 norms between the images and
their manually computed down-sampled representation. During training, separated
optimiser are implemented to decrease the losses independently.

As the hidden layers can be modulated in size using the script parameters, they
have to be identical from a call to another on a specific data-set and trained
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

## References

[1] T. Karras, T. Aila, S. Laine, and J. Lehtinen. Progressive growing of gans for improved quality, stability, and variation. arxiv:1710.10196, 2017
