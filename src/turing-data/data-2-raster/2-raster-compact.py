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
import matplotlib.image as ml_image

##
##  script - argument and parameter
##

# create argument parser #
ml_apar = argparse.ArgumentParser()

# argument directive #
ml_apar.add_argument( '-i', '--image'  , required=True, type=str, help='dataset images directory' )
ml_apar.add_argument( '-o', '--dataset', required=True, type=str, help='dataset path' )

##
##  script - image exportation
##

def ml_export( ml_data, ml_path ):

    # create output stream #
    with open( ml_path, 'ab' ) as ml_file:

        # export array in file #
        numpy.array( ml_data, dtype=numpy.uint8 ).tofile( ml_file )

##
##  script - image normalisation
##

def ml_normalise( ml_data ):

    # check image shape #
    if ( len( ml_data.shape ) == 3 ):

        # check image size #
        if ( ml_data.shape[2] == 3 ) or ( ml_data.shape[2] == 4 ):

            # compute luminance #
            ml_data = 0.2126 * ml_data[:,:,0] + 0.7152 * ml_data[:,:,1] + 0.0722 * ml_data[:,:,2]

        else:

            # check image size #
            if ( ml_data.shape[2] != 1 ):

                # send message #
                sys.exit( 'turing : error : unknown image format' )

    else:

        # check image shape #
        if ( len( ml_data.shape ) != 2 ):

            # send message #
            sys.exit( 'turing : error : unknown image format' )

    # normalise on [0,255] range #
    ml_data = numpy.multiply( ml_data, 255.0 )

    # return image #
    return ml_data

##
##  script - main function
##

# read argument and parameter #
ml_args = ml_apar.parse_args()

# enumerate file in dataset directory #
for ml_file in os.listdir( ml_args.image ):

    # display information #
    print( 'turing : compacting ' + ml_file + '...' )

    # load portable network graphics image #
    ml_data = ml_image.imread( ml_args.image + '/' + ml_file )

    # normalise image format #
    ml_data = ml_normalise( ml_data )

    # export image in dataset file #
    ml_export( ml_data, ml_args.dataset )

# display informatioin #
print( 'turing : done' )

