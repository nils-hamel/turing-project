# Overview

This repository contains the researches codes, analysis and results of deep learning applied to 2-dimensional, 3-dimensional, 4-dimensional and 5-dimensional objects in the domain of the geodesy and geography. The goal of the conducted research is to determine at which extent deep learning can be applied to address the challenges that appears in such domain.

More precisely, the researches published on this repository study which neural network topologies have the best training and generalisation performance addressing specific tasks in the domain of geodesy and geography. It follows that this repository holds many different codes and trials and the published researches are often driven under a systematic approach, showing both good and bad results.

# Turing Project

The _Turing Project_ can be seen as three parts : the first one are the ensemble of the considered data-sets used for neural networks analysis. The second part contains the neural networks implementation script that are tested and analysed. The last part are
the results and their subsequent analysis.

## Turing Project : Data-Sets

Two main type of digital objects are considered in this approach of deep learning applied to geodesy and geography. The first one are the 2-dimensional and 3-dimensional raster containing greyscale of color images. The second type of objects are 3-dimensional or 4-dimensional greyscale or color 3D models sampled over voxel grids. Of course, more complex combination of those objects are also considered for deeper analysis.

### Turing Project : Image Raster

Different image data-sets are considered in the conducted research to also analyse the way neural network are able learn different type of representation. The considered data-sets are presented here through their own page :

* 64x64x3-geneva-2009 ([v1.0](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-geneva-2009.md))
* 64x64x3-geneva-2011 ([v1.0](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-geneva-2011.md))

# Copyright and License

**turing-project** - Nils Hamel <br >
Copyright (c) 2016-2017 DHLAB, EPFL

The codes are licensed under the terms of the GNU GPLv3. The documents and illustrations are licensed under the CC BY-SA 4.0.

# Dependencies

The _turing-project_ comes with the following package dependencies (Ubuntu 16.04 LTS) :

* bash
* python3

and with the following python3 modules :

* numpy
* sys
* os
* matplotlib
* argparse
* tensorflow ([installation instructions](https://www.tensorflow.org/install/))
* random

Note that the python scripts included in this directory are designed for python3. Nevertheless, most of the time the scripts can simply be converted to python2.
