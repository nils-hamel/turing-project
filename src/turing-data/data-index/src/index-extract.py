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
ml_apar.add_argument( '-m', '--mode'   , type=str, help='script mode'  )
ml_apar.add_argument( '-d', '--dataset', type=str, help='dataset path' )
ml_apar.add_argument( '-w', '--width'  , type=int, help='raster width' )
ml_apar.add_argument( '-r', '--raster' , type=str, help='raster path'  )
ml_apar.add_argument( '-c', '--count'  , type=int, help='raster count' )

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
##  script - dataset exportation
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
ml_data = ml_raster_import( ml_args.dataset, ml_args.width )

# check script mode #
if ( ml_args.mode == 'full' ):

    # parsing dataset #
    for ml_parse in range( ml_data.shape[0] ):

        # display information #
        print( 'turing : export raster ' + str( ml_parse ) + ' ...' )

        # export raster #
        ml_raster_export( ml_data[ml_parse], ml_args.raster + '/raster-{:06d}.ras'.format( ml_parse ) )

    # display information #
    print( 'turing : done' )

elif ( ml_args.mode == 'sample' ):

    # parsing dataset #
    for ml_parse in range( ml_args.count ):

        # create random index #
        ml_index = random.randint( 0, ml_data.shape[0] - 1 )

        # display information #
        print( 'turing : export raster ' + str( ml_index ) + ' ...' )

        # export raster #
        ml_raster_export( ml_data[ml_parse], ml_args.raster + '/raster-{:06d}.ras'.format( ml_index ) )

    # display information #
    print( 'turing : done' )

