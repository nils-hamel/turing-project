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

import numpy
import sys
import os
import matplotlib.image as image

##
##  script - dataset import
##

def ml_data_import( ml_path, ml_width ):

    # check consistency #
    if ( os.path.exists( ml_path ) == False ):

        # send message #
        sys.exit( 'turing : error : unable to access dataset' )

    # dataset input stream #
    with open( ml_path, 'rb' ) as ml_file:

        # import dataset bytes #
        ml_byte = ml_file.read( os.path.getsize( ml_path ) )

    # convert bytes to numpy array #
    ml_data = numpy.frombuffer( ml_byte, dtype=numpy.uint8 )

    # return dataset #
    return ml_data.reshape( -1, ml_width, ml_width, ml_width )

##
##  script - dataset format
##

def ml_data_format_float( ml_data ):

    # convert dataset values type #
    return ml_data.astype( numpy.float32 )

def ml_data_format_uint8( ml_data ):

    # round dataset values #
    ml_data = numpy.around( ml_data )

    # convert dataset values type #
    return ml_data.astype( numpy.uint8 )

def ml_data_format_central( ml_data ):

    # renormalisation of dataset values : [0,1] to [-1,+1] #
    return numpy.multiply( ml_data - 0.5, 2.0 )

def ml_data_format_central_invert( ml_data ):

    # renormalisation of dataset values : [-1,+1] to [0,1] #
    return numpy.multiply( ml_data + 1.0, 0.5 )

##
##  script - dataset shuffle
##

def ml_data_random( ml_data ):

    # create index array #
    ml_index = numpy.arange( ml_data.shape[0] )

    # shuffle index array #
    numpy.random.shuffle( ml_index )

    # return index array #
    return ml_index

def ml_data_shuffle( ml_data, ml_index ):

    # check consistency #
    if ( ml_data.shape[0] != ml_index.shape[0] ):

        # send message #
        sys.exit( 'turing : errro : unable to shuffle data with index' )

    # create copy of the dataset #
    ml_copy = numpy.copy( ml_data )

    # parsing dataset #
    for ml_parse in range( ml_data.shape[0] ):

        # assign randomised element #
        ml_data[ml_parse] = ml_copy[ml_index[ml_parse]]

    # return randomised dataset #
    return ml_data

##
##  script - dataset range
##

def ml_data_split( ml_data, ml_proportion, ml_batch_size ):

    # compute nearest batch-size multiple #
    ml_index = int( int( ml_data.shape[0] * ml_proportion ) / ml_batch_size ) * ml_batch_size

    # compute upper boundary #
    ml_bound = ml_index + ml_batch_size

    # return training, training-loss and validation loss sub-dataset #
    return ml_data[:ml_index], numpy.copy( ml_data[:ml_batch_size] ), numpy.copy( ml_data[ml_index:ml_bound] )

def ml_data_range( ml_data, ml_start, ml_stop ):

    # check consistency #
    if ( ( ml_data.shape[0] < ml_stop ) or ( ml_start > ml_stop ) ):

        # send message #
        sys.exit( 'turing : error : selection out of range' )

    # return selected sub-dataset #
    return ml_data[ml_start:ml_stop]

##
##  script - dataset minibatch
##

def ml_data_batch_count( ml_data, ml_batch_size ):

    # compute and return mini-batch count #
    return int( ml_data.shape[0] / ml_batch_size )

def ml_data_batch( ml_data, ml_batch_size, ml_batch ):

    # check consistency #
    if ( ml_batch > ml_data_batch_count( ml_data, ml_batch_size ) ):

        # send message #
        sys.exit( 'turing : error : batch out of range' )

    # compute batch offset #
    ml_offset = ml_batch_size * ml_batch

    # compute upper boundary #
    ml_bound = ml_offset + ml_batch_size

    # return mini-batch #
    return ml_data[ml_offset:ml_bound]

##
##  script - raster manipulation
##

def ml_data_raster_save( ml_raster, ml_path ):

    # raster output stream #
    with open( ml_path, 'wb' ) as ml_file:

        # export raster #
        numpy.array( ml_raster, dtype=numpy.uint8 ).tofile( ml_file )

##
##  script - vector manipulation
##

def ml_data_vector_save( ml_vector, ml_path ):

    # export vector #
    numpy.savetxt( ml_path, ml_vector )

def ml_data_vector_load( ml_path ):

    # load and return vector #
    return numpy.loadtxt( ml_path )

