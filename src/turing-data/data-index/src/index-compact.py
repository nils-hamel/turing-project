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
import matplotlib.image as ml_image

##
##  script - argument and parameter
##

# create argument parser #
ml_apar = argparse.ArgumentParser()

# argument directive #
ml_apar.add_argument( '-r', '--raster' , type=str, help='raster path'  )
ml_apar.add_argument( '-d', '--dataset', type=str, help='dataset path' )

# read argument and parameter #
ml_args = ml_apar.parse_args()

##
##  script - raster i/o operation
##

def ml_raster_import( ml_path ):

    # check consistency #
    if ( not os.path.exists( ml_path ) ):

        # send message #
        sys.exit( 'turing : error : unable to access raster' )

    # retrieve raster size #
    ml_size = os.path.getsize( ml_path )

    # compute raster width #
    ml_width = int( round( ml_size ** ( 1.0 / 3.0 ) ) )

    # import raster data #
    with open( ml_path, 'rb' ) as ml_file:

        # read raster bytes #
        ml_byte = ml_file.read( ml_size )

    # convert to numpy array #
    ml_data = numpy.frombuffer( ml_byte, dtype=numpy.uint8 )

    # return raster array #
    return ml_data.reshape( ml_width, ml_width, ml_width )

def ml_raster_export( ml_data, ml_path ):

    # create output stream #
    with open( ml_path, 'ab' ) as ml_file:

        # export raster array #
        numpy.array( ml_data, dtype=numpy.uint8 ).tofile( ml_file )    

##
##  script - main function
##

# enumerate file in dataset directory #
for ml_file in os.listdir( ml_args.raster ):

    # check file extension #
    if ( ml_file.endswith( ".ras" ) ):

        # display information #
        print( 'turing : compacting ' + ml_file + ' ...' )

        # import and export raster in dataset #
        ml_raster_export( ml_raster_import( ml_args.raster + '/' + ml_file ), ml_args.dataset )

    else:

        # display information #
        print( 'turing : reject ' + ml_file )

# display information #
print( 'turing : done' )

