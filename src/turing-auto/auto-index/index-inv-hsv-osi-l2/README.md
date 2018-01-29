## Auto-Encoder - Index - inv-hsv-osi-l2

This directory contains the implementation of a single hidden layer auto-encoder
for index rasters. Index rasters are understood as binary information distributed
on a three-dimensional homogeneous grid.

The topology of the auto-encoder is as follows : the network starts with the
input layer that has as many neurons as the input raster has elements. The input
layer is connected to the hidden layer. The hidden layer can be modulated in
size. The hidden layer is then connected to the output layer that has the same
size as the input layer. Both hidden and output layers come with a sigmoid as
activation function.

The cost function of the network is a simple L2 norm between the input and the
output layers. The global minimum of the cost function is then reached as the
input and output layers contain the same index raster.

As the hidden layer can be modulated in size using the script parameters, it has
to be identical from a call to another on a specific data-set and trained
network. Loading a network that has been trained with a different hidden size
will raise an error.

The input layer is fed using index raster in single precision. In principle, the
input layer should only contains values equal to zero or one. The output layer
is expected in the same format but a rounding function is mandatory to end up
with an output layer containing a valid raster for exportation.

The network implementation script provides different modes. The first mode
corresponds to the network training and retraining process. The second mode
allows to auto-encode rasters of the data-set by specifying the range of the
rasters to consider. The next mode allows to encode a specified range of rasters
of the data-set in the auto-encoder latent space. The output of this mode is
then a set of vectors. The last mode allows to read a set of vectors interpreted
as the auto-encoder latent values to produce to corresponding index rasters.

## Copyright and License

**turing-project** - Nils Hamel <br >
Copyright (c) 2016-2018 DHLAB, EPFL

The codes are licensed under the terms of the GNU GPLv3.
