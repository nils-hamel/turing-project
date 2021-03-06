# Overview

This repository contains the researches codes, analysis and results of deep learning applied to 2-dimensional, 3-dimensional, 4-dimensional and 5-dimensional objects in the domain of the geodesy and geography. The goal of the conducted research is to determine at which extent deep learning can be applied to address the challenges that appears in such domain.

More precisely, the researches published on this repository study which neural network topologies have the best training and generalisation performance addressing specific tasks in the domain of geodesy and geography. It follows that this repository holds many different codes and trials and the published researches are often driven under a systematic approach, showing both good and bad results.

# Turing Project

The _Turing Project_ can be split in three main parts : the first one are the ensemble of the considered data-sets used for neural networks analysis. The second part contains the neural networks implementation script that are tested and analysed. The last part are
the results and their subsequent analysis.

## Turing Project : Data-Sets

Two main type of digital objects are considered in this approach of deep learning applied to geodesy and geography. The first one are the 2-dimensional and 3-dimensional raster containing greyscale of color images. The second type of objects are 3-dimensional or 4-dimensional greyscale or color 3D models sampled over voxel grids. Of course, more complex combination of those objects are also considered for deeper analysis.

### Data-Sets : Image Rasters

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

### Data-Sets : Index Rasters

In addition to image data-set, index rasters are also considered. Index raster are 3D volumetric data of the form of voxel grids. These data-sets are used to determine in which extend 3D data can also be learned by neural networks. Each of the following page give a presentation of a data-set :

* [64x64x64-geneva-2005](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x64-geneva-2005.md)
* [32x32x32-geneva-2005](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/32x32x32-geneva-2005.md)
* [64x64x64-geneva-2009](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x64-geneva-2009.md)
* [32x32x32-geneva-2009](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/32x32x32-geneva-2009.md)
* [64x64x64-europe-2000](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/64x64x64-europe-2000.md)
* [32x32x32-europe-2000](https://github.com/nils-hamel/turing-project/blob/master/doc/dataset/32x32x32-europe-2000.md)

## Turing Project : Neural Networks

Different type of neural networks are analysed in this approach of deep learning applied to geodesy and geography. The first type of neural networks are the auto-encoder used to determine the ability of neural networks to reduce and generalise sets of data.

### Neural Networks : Auto-Encoders

Auto-encoders are used to determine in which extend neural networks are able to learn and generalise specific data-sets in the domain of geodesy and geography. The goal of the study of auto-encoders is to verify their ability to understand the grammar of specific places and environments mainly from the point of view of interpretation and completion.

#### Auto-encoders for image raster

Auto-encoders for image raster are applied, trained and validated here on two dimensional data grids.

Single-layer auto-encoders for image raster :

* [inv-hsv-osi-l2 : 1-layer sigmoid-l2 auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-inv-hsv-osi-l2)
* [icv-hhv-ohi-l2 : 1-layer asinh-l2 auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-icv-hhv-ohi-l2)
* [icv-htv-oti-l2 : 1-layer tanh-l2 auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-icv-htv-oti-l2)
* [icv-hrv-ohi-l2 : 1-layer relu/asinh-l2 auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-icv-hrv-ohi-l2)
* [icv-hrv-oti-l2 : 1-layer relu/tanh-l2 auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-icv-hrv-oti-l2)
* [inv-hsv-osi-ce : 1-layer sigmoid-ce auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-inv-hsv-osi-ce)

Single-convolution auto-encoders for image raster :

* [inv-c5rv-d5si-l2 : 1-convolution relu-sigmoid-l2 auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-inv-c5rv-d5si-l2)

Asymmetrical single-convolution and single-layer auto-encoders for image raster :

* [inv-c5rv-hsv-osi-l2 : 1-convolution/1-layer relu-sigmoid-l2 auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-inv-c5rv-hsv-osi-l2)

Double-layer auto-encoders for image raster :

* [inv-hsv-osi-l2-hsv-osi-l2 : 2-layer sigmoid-l2 dual-loss auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-inv-hsv-osi-l2-hsv-osi-l2)

Three-layer auto-encoders for image raster :

* [inv-hsv-hsv-hsv-osi-l2 : 3-layer sigmoid-l2 auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-inv-hsv-hsv-hsv-osi-l2)
* [icv-hhv-hhv-hhv-ohi-l2 : 3-layer asinh-l2 auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-icv-hhv-hhv-hhv-ohi-l2)
* [icv-htv-htv-htv-oti-l2 : 3-layer tanh-l2 auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-icv-htv-htv-htv-oti-l2)
* [inv-hsv-hsv-hsv-osi-ce : 3-layer sigmoid-ce auto-encoder for image raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-image/image-inv-hsv-hsv-hsv-osi-ce)

#### Auto-encoders for index raster

Auto-encoders for index raster are applied, trained and validated here on three dimensional data grids.

Single-layer auto-encoders for index raster :

* [inv-hsv-osi-l2 : 1-layer sigmoid-l2 auto-encoder for index raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-auto/auto-index/index-inv-hsv-osi-l2)

### Neural Networks : Trans-Coders

Trans-coders are used to determine in which extend neural networks are able to learn and generalise on specific trans-coding task applied in the domain of geodesy and geography. Trans-coders are studied to analyse their abilities to understand sufficiently the grammar of specific places and environments to compute trans-coded representation such as super-scaled models.

#### Trans-Coders for index raster

Tans-coders for index raster are applied, trained and validated here on three dimensional data grids.

Single-layer trans-coder for index raster :

* [inv-hsv-osv-l2  : 1-layer sigmoid-l2 trans-coder for index raster](https://github.com/nils-hamel/turing-project/tree/master/src/turing-tran/tran-index/index-inv-hsv-osv-l2)

## Turing Project : Researches

Researches are linked to the implemented neural networks and the data-sets on which they are applied. The following section give access to the research performed under the _Turing Project_.

### Researches : Auto-Encoders for image raster

The following research presents the result of the analysis of the training and
generalisation results of auto-encoders applied on 2-dimensional image raster.
The implementation of the auto-encoders allows to modulate the sizes of the
hidden layers in order to perform systematic analysis of the networks performances.

Research on single-layer auto-encoders :

* [inv-hsv-osi-l2 training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/8a02301cb3b9f308.md)
* [icv-hhv-ohi-l2 training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/7002de414bd53e16.md)
* [icv-htv-oti-l2 training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/1dc3c6eca32c689f.md)
* [icv-hrv-ohi-l2 training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/c13e876576c6ecb0.md)
* [icv-hrv-oti-l2 training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/72d52f8ddd4b6619.md)
* [inv-hsv-osi-ce training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/f343816a10b17cc9.md)

Research on single-convolution auto-encoders :

* [inv-c5rv-d5si-l2 training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/5c6f602bcd5ba11e.md)

Research on  single-convolution and single-layer auto-encoders :

* [inv-c5rv-hsv-osi-l2 training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/7ac258f1f4e6c707.md)

Research on double-layer auto-encoders :

* [inv-hsv-osi-l2-hsv-osi-l2 training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/9cad7ad9b2daf12c.md)

Research on three-layer auto-encoders :

* [inv-hsv-hsv-hsv-osi-l2 training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/ce026ed0ca472cbb.md)
* [icv-hhv-hhv-hhv-ohi-l2 training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/8f7a9e0546192e36.md)
* [icv-htv-htv-htv-oti-l2 training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/752ab4601e1589d8.md)
* [inv-hsv-hsv-hsv-osi-ce training and validation](https://github.com/nils-hamel/turing-project/blob/master/doc/research/c51df6368f404cc5.md)

# Copyright and License

**turing-project** - Nils Hamel <br >
Copyright (c) 2016-2018 DHLAB, EPFL

The codes are licensed under the terms of the GNU GPLv3. The documents and illustrations are licensed under the CC BY-SA 4.0. In addition, the data-sets content and results illustrations contain elements under subsequent copyrights. Please refer to the data-sets pages for more information.

# Dependencies

The _turing-project_ comes with the following package dependencies (Ubuntu 16.04 LTS) :

* bash
* python3
* bc
* imagemagick

and the following external dependencies :

* eratosthene-suite (v0.3.7)

and with the following python3 modules :

* numpy
* sys
* os
* matplotlib
* argparse
* tensorflow ([installation instructions](https://www.tensorflow.org/install/))
* random

Note that the python scripts included in this directory are designed for python3. Nevertheless, most of the time the scripts can simply be converted to python2.
