# Overview

This repository contains the researches codes, analysis and results of deep learning applied to 2-dimensional, 3-dimensional, 4-dimensional and 5-dimensional objects in the domain of the geodesy and geography. The goal of the conducted research is to determine at which extent deep learning can be applied to address the challenges that appears in such domain.

More precisely, the researches published on this repository study which neural network topologies have the best training and generalisation performance addressing specific tasks in the domain of geodesy and geography. It follows that this repository holds many different codes and trials and the published researches are often driven under a systematic approach, showing both good and bad results.

# Turing Project

The _Turing Project_ can be split in three main parts : the first one are the ensemble of the considered data-sets used for neural networks analysis. The second part contains the neural networks implementation script that are tested and analysed. The last part are
the results and their subsequent analysis.

## Turing Project : Data-Sets

Two main type of digital objects are considered in this approach of deep learning applied to geodesy and geography. The first one are the 2-dimensional and 3-dimensional raster containing greyscale of color images. The second type of objects are 3-dimensional or 4-dimensional greyscale or color 3D models sampled over voxel grids. Of course, more complex combination of those objects are also considered for deeper analysis.

### Data-Sets : Image Raster

Different image data-sets are considered in the conducted research to also analyse the way neural network are able learn different type of representation. The considered data-sets are presented here through their own page :

* [64x64x3-geneva-2009](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-geneva-2009.md)
* [64x64x3-geneva-2011](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-geneva-2011.md)
* [64x64x3-geneva-2016](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-geneva-2016.md)
* [64x64x3-geneva-oblique-2013](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-geneva-oblique-2013.md)
* [64x64x3-paris-louvre-2017](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-paris-louvre-2017.md)
* [64x64x3-venezia-2004](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-venezia-2004.md)
* [64x64x3-venezia-2010](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-venezia-2010.md)
* [64x64x3-venezia-2014](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-venezia-2014.md)
* [64x64x3-venezia-campanile-2016](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-venezia-campanile-2016.md)
* [64x64x3-venezia-palazzo-ducale-2016](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-venezia-palazzo-ducale-2016.md)
* [64x64x3-venezia-piazza-2016](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-venezia-piazza-2016.md)
* [64x64x3-venezia-san-giacometto-2016](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-venezia-san-giacometto-2016.md)
* [64x64x3-venezia-san-marco-2016](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x3-venezia-san-marco-2016.md)

## Turing Project : Neural Networks

Different type of neural networks are analysed in this approach of deep learning applied to geodesy and geography. The first type of neural networks are the auto-encoder used to determine the ability of neural networks to reduce and generalise sets of data.

### Neural Networks : Auto-Encoder

Auto-encoders are used to determine in which extend neural networks are able to learn and generalise specific data-sets in the domain of geodesy and geography. The goal of the study of auto-encoders is to verify their ability to understand the grammar of specific places and environments mainly from the point of view of interpretation and completion.

Several _academic_ auto-encoders are considered and listed here :

* [inv-hsv-osi-l2 : 1-layer auto-encoder for image raster](https://github.com/nils-hamel/turing-project/blob/master/src/turing-auto/auto-inv-hsv-osi-l2/README.md)
* [inv-hsv-hsv-hsv-osi-l2 : 3-layer auto-encoder for image raster](https://github.com/nils-hamel/turing-project/blob/master/src/turing-auto/auto-inv-hsv-hsv-hsv-osi-l2/README.md)

## Turing Project : Researches

Researches are linked to the implemented neural networks and the data-sets on which they are applied. The following section give access to the research performed under the _Turing Project_.

### Researches : Auto-Encoders for image raster

The following research presents the result of the analysis of the training and generalisation results of auto-encoders applied on 2-dimensional image raster. The implementation of the auto-encoders allows to modulate the sizes of the hidden layers in order to perform systematic analysis of the networks performances.

* [Generalisation analysis according to data-sets and hidden layer size of single layer auto-encoder](https://github.com/nils-hamel/turing-project/blob/master/doc/research/research-auto-single-training.md)

# Copyright and License

**turing-project** - Nils Hamel <br >
Copyright (c) 2016-2017 DHLAB, EPFL

The codes are licensed under the terms of the GNU GPLv3. The documents and illustrations are licensed under the CC BY-SA 4.0. In addition, the data-sets content and results illustrations contain elements under subsequent copyrights. Please refer to the data-sets pages for more information.

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
