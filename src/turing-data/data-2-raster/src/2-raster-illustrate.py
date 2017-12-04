#!/usr/bin/env python3

## turing project
##
##     Nils Hamel - nils.hamel@bluewin.ch
##     Copyright (c) 2016-2017 DHLAB, EPFL
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import numpy
import os
import sys
import random
import matplotlib.image as ml_image

##
##  script - argument and parameter
##

# create argument parser #
ml_apar = argparse.ArgumentParser()

# argument directive #
ml_apar.add_argument( '-i', '--dataset', type=str, help='dataset path'    )
ml_apar.add_argument( '-w', '--size'   , type=int, help='dataset witdh'   )
ml_apar.add_argument( '-o', '--image'  , type=str, help='image path'      )
ml_apar.add_argument( '-c', '--count'  , type=int, help='selection count' )

# read argument and parameter #
ml_args = ml_apar.parse_args()

##
##  script - image exportation
##

def ml_export( ml_data, ml_path ):

    # create 3-layers matrix #
    ml_data = numpy.reshape( numpy.repeat( ml_data[:, :, numpy.newaxis], 3, axis=2 ), newshape=( ml_data.shape[0], ml_data.shape[1], 3 ) )

    # export image #
    ml_image.imsave( ml_path, ml_data )

##
##  script - dataset importation
##

def ml_import( ml_path, ml_resolution ):

    # check consistency #
    if ( not os.path.exists( ml_path ) ):

        # send message #
        sys.exit( 'turing : error : unable to access dataset' )

    # create input stream #
    with open( ml_path, 'rb' ) as ml_file:

        # read bytes #
        ml_byte = ml_file.read( os.path.getsize( ml_path ) )

    # convert data #
    ml_data = numpy.frombuffer( ml_byte, dtype=numpy.uint8 )

    # convert data #
    ml_data = ml_data.astype( numpy.float32 )

    # renormalise on [0,1] range #
    ml_data = numpy.multiply( ml_data, 1.0 / 255.0 )

    # reshape array #
    ml_data = ml_data.reshape( -1, ml_resolution, ml_resolution )

    # return array #
    return ml_data

##
##  script - main function
##

# import data #
ml_data = ml_import( ml_args.dataset, ml_args.size )

# selecting random dataset image #
for ml_parse in range( ml_args.count ):

    # create random index #
    ml_index = random.randint( 0, ml_data.shape[0] - 1 )

    # export selected image #
    ml_export( ml_data[ml_index], ml_args.image + '/image-{:06d}.png'.format( ml_parse ) )
    
