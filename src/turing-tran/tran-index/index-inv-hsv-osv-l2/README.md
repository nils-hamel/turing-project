## Trans-Coder - Index - inv-hsv-osv-l2

This directory contains the implementation of a single hidden layer trans-coder
for index rasters. Index rasters are understood as binary information distributed
on a three-dimensional homogeneous grid.

The implemented network topology starts with an input layer that is adapted, in
terms of number of units, to the size of the input index rasters. This layer
is followed by a dense hidden layer activated by a sigmoid function. The hidden
layer is connected to the output layer that have as many units as the output
index rasters have spatial-cells.

During training, the trans-coded index are compared to target index through a
simple L2 norm. The target index are used to explain how the input index have
to be trans-coded.

As the hidden layer can be modulated in size using the script parameters, it has
to be identical from a call to another on a specific data-set and trained
network. Loading a network that has been trained with a different hidden size
will raise an error.

The input layer is fed using index raster in single precision. In principle, the
input layer should only contains values equal to zero or one. The output layer
is expected in the same format but a rounding function is mandatory to end up
with an output layer containing a valid raster for exportation.

The implementation script provides two main modes. The first mode are used to
train and re-train the network on a set of two data-set (corresponding to the
input index and the target index). The second mode is used to perform trans-coding
on the input data-set in order to export the trans-coded results has a trained
network is available.

## Copyright and License

**turing-project** - Nils Hamel <br >
Copyright (c) 2016-2018 DHLAB, EPFL

The codes are licensed under the terms of the GNU GPLv3.
