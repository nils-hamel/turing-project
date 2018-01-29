#!/usr/bin/env python3

## turing project
##
##     Nils Hamel - nils.hamel@bluewin.ch
##     Copyright (c) 2016-2018 DHLAB, EPFL
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

##
##  script - argument and parameter
##

# create argument parser #
ml_apar = argparse.ArgumentParser()

# argument directive #
ml_apar.add_argument( '-i', '--input' , type=str, help='input dataset'  )
ml_apar.add_argument( '-w', '--width' , type=int, help='raster width'   )
ml_apar.add_argument( '-o', '--output', type=str, help='output dataset' )

# read argument and parameter #
ml_args = ml_apar.parse_args()

##
##  script - raster exportation
##

def ml_raster_export( ml_data, ml_path ):

    # create output stream #
    with open( ml_path, 'ab' ) as ml_file:

        # export raster array #
        numpy.array( ml_data, dtype=numpy.uint8 ).tofile( ml_file )

##
##  script - raster down-sampling
##

def ml_raster_downsample( ml_raster ):

    # create downsampled raster #
    ml_down = numpy.zeros( [ int( ml_raster.shape[0] / 2 ), int( ml_raster.shape[1] / 2 ), int( ml_raster.shape[2] / 2 ) ], dtype=numpy.uint8 )

    # parsing dimension #
    for ml_x in range( ml_raster.shape[0] ):

        # parsing dimension #
        for ml_y in range( ml_raster.shape[1] ):

            # parsing dimension #
            for ml_z in range( ml_raster.shape[2] ):

                # check element value #
                if ( ml_raster[ml_x, ml_y, ml_z] == 1 ):

                    # assign down-sampled element #
                    ml_down[int(ml_x/2),int(ml_y/2),int(ml_z/2)] = 1

    # return down-sampled raster #
    return ml_down

##
##  script - dataset importation
##

def ml_raster_import( ml_path, ml_width ):

    # check consistency #
    if ( not os.path.exists( ml_path ) ):

        # send message #
        sys.exit( 'turing : error : unable to access dataset' )

    # create input stream #
    with open( ml_path, 'rb' ) as ml_file:

        # import bytes #
        ml_byte = ml_file.read( os.path.getsize( ml_path ) )

    # convert to numpy array #
    ml_data = numpy.frombuffer( ml_byte, dtype=numpy.uint8 )

    # return dataset #
    return( ml_data.reshape( -1, ml_width, ml_width, ml_width ) )

##
##  script - main function
##

# import dataset #
ml_data = ml_raster_import( ml_args.input, ml_args.width )

# parsing dataset #
for ml_parse in range( ml_data.shape[0] ):

    # display information #
    print( 'turing : down-sample raster ' + str( ml_parse ) + ' ...' )

    # down-sample raster #
    ml_down = ml_raster_downsample( ml_data[ml_parse] )

    # export down-sampled raster #
    ml_raster_export( ml_down, ml_args.output )

# display information #
print( 'turing : done' )

