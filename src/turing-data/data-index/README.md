## Data Processing : Index Raster

This directory contains the sources of the 3D raster processing script. 3D raster
are defined here as three dimensional array containing binary value to indicate
the presence of elements.

## Script Overview

The following script are used to compose data-set, extract elements from data-set
remote server query tool and 3D raster content display. The following sub-sections
offer a brief overview of each script.

### Script : index-show

This script is used to display the content of a specified 3D raster. The raster
is converted by the script in a set of point that is displayed in a 3D frame.
The script allows both to view and export the produced figure.

### Script : index-compact

This script is used to compose data-set from a collection of 3D rasters. The
script allows to reads the 3D raster collection to produce a single data-set
file containing all the elements of the collection.

### Script : index-extract

This script allows to extract all or samples of the elements contained in a
specified data-set. The script reads the elements from the data-set file and
export each element in a specific 3D raster file.

Two extraction modes are available : the _full_ mode that performs an extraction
of all elements contained in the specified data-set file and the _sample_ mode
that makes a random selection of _N_ element of the data-set, _N_ having to be
specified in this case.

### Script : index-query

This script is used to compose collection of 3D raster. It requires the availability
of a remote _eratosthene_ server in order to operate.

Based on a provided spatial components of index, the script asks the _eratosthene-suite_
tool to enumerates all sub-index up to a specified depth. A limit value can be
specified in order to only keep raster with a certain amount of points.

## Copyright and License

**turing-project** - Nils Hamel <br >
Copyright (c) 2016-2018 DHLAB, EPFL

The codes are licensed under the terms of the GNU GPLv3.
