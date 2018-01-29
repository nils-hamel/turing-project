## Data Processing : Image Raster

This directory contains the sources of images raster processing scripts. Image
rasters are understood here as simple red green and blue pixels matrix. It
follows that any image raster data-set is a collection of colorimetric pixels
matrix.

## Script Overview

Several script are implemented to help creation and extraction of data-sets
composed of image rasters. The following sub-sections offer a brief overview of
each script.

### image-photo

This bash script is used to extract sub-image of a given resolution from a set
of photography. This allows to randomly crop sub-region of the photography of
the set in order to compose a data-set larger than the original photography
collection.

The script allows to specify the size of the data-set image raster and to
modulate the number of sub-image extraction by specifying the scale and the
amount of crop to perform on each scale.

### image-compact

This script, implemented in python, allows to read an image collection from a
directory to compact them into a single data-set file. This type of file is
usually expected by the implemented neural network of this repository.

The script expects and image collection containing colorimetric images having
the same resolution. The script also ensure that the color component are packed
in 24 bits (uint8).

### image-extract

This script, implemented in python, allows to extract the content of a data-set
file in a specified directory. For each extracted element of the data-set, the
script create a portable network graphic file in which the element is exported.

Two extraction modes are available : the _full_ mode that performs an extraction
of all elements contained in the specified data-set file and the _sample_ mode
that makes a random selection of _N_ element of the data-set, _N_ having to be
specified in this case.

## Copyright and License

**turing-project** - Nils Hamel <br >
Copyright (c) 2016-2018 DHLAB, EPFL

The codes are licensed under the terms of the GNU GPLv3.
