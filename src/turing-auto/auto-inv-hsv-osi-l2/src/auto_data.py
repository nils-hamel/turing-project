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

def ml_data_import( ml_path, ml_size ):

    # check consistency #
    if ( os.path.exists( ml_path ) == False ):

        # send message #
        sys.exit( 'turing : error : unable to access dataset' )

    # dataset input stream #
    with open( ml_path, 'rb' ) as ml_file:

        # import bytes #
        ml_byte = ml_file.read( os.path.getsize( ml_path ) )

    # convert to numpy array #
    ml_data = numpy.frombuffer( ml_byte, dtype=numpy.uint8 )

    # floating-point array #
    ml_data = ml_data.astype( numpy.float32 )

    # renormalise on range [0,1] #
    ml_data = numpy.multiply( ml_data, 1.0 / 255.0 )

    # return dataset #
    return ml_data.reshape( -1, ml_size, ml_size, 3 )

##
##  script - dataset format
##

def ml_data_format_y( ml_data ):
    
    # return converted data #
    return 0.2126 * ml_data[:,:,:,0] + 0.7152 * ml_data[:,:,:,1] + 0.0722 * ml_data[:,:,:,2]

def ml_data_format_central( ml_data ):

    # return centralised dataset - normalisation on [-1,+1] #
    return numpy.multiply( ml_data - 0.5, 2.0 )

def ml_data_format_central_invert( ml_data ):

    # return inverted centralised dataset #
    return numpy.multiply( ml_data + 1.0, 0.5 )

##
##  script - dataset shuffle
##

def ml_data_random( ml_data ):

    # create index array #
    ml_index = numpy.arange( ml_data.shape[0] )

    # randomise index array #
    numpy.random.shuffle( ml_index )

    # return randomised index #
    return ml_index

def ml_data_shuffle( ml_data, ml_index ):

    # check consistency #
    if ( ml_data.shape[0] != ml_index.shape[0] ):

        # send message #
        sys.exit( 'turing : error : vector must have the same size' )

    # create data copy #
    ml_copy = numpy.copy( ml_data )

    # parsing index #
    for ml_parse in range( ml_data.shape[0] ):

        # assign element #
        ml_data[ml_parse] = ml_copy[ml_index[ml_parse]]

    # return dataset #
    return ml_data

##
##  script - dataset range
##

def ml_data_split( ml_data, ml_proportion, ml_batch_size ):

    # compute splitting index #
    ml_index = int( ml_data.shape[0] * ml_proportion )

    # compute nearest batch multiple #
    ml_index = int( ml_index / ml_batch_size ) * ml_batch_size

    # return splitted dataset #
    return numpy.copy( ml_data[:ml_index] ), numpy.copy( ml_data[:ml_data.shape[0] - ml_index] ), numpy.copy( ml_data[ml_index:] )

def ml_data_range( ml_data, ml_start, ml_stop ):

    # check range #
    if ( ( ml_data.shape[0] < ml_stop ) or ( ml_start > ml_stop ) ):

        # send message #
        sys.exit( 'turing : error : selection out of range' )

    # return selected range #
    return numpy.copy( ml_data[ml_start:ml_stop] )

##
##  script - dataset minibatch
##

def ml_data_batch_count( ml_data, ml_batch_size ):

    # return minibatch count #
    return int( ml_data.shape[0] / ml_batch_size )

def ml_data_batch( ml_data, ml_batch_size, ml_index ):

    # check consistency #
    if ( ml_index > ml_data_batch_count( ml_data, ml_batch_size ) ):

        # send message #
        sys.exit( 'turing : error : batch index out of range' )

    # compute offset #
    ml_offset = ml_batch_size * ml_index

    # compute boundary #
    ml_bound = ml_offset + ml_batch_size

    # return minibatch #
    return ml_data[ ml_offset : ml_bound ]

##
##  script - image manipulation
##

def ml_data_image_save( ml_image, ml_path ):

    # create 3-layers matrix #
    ml_image = numpy.reshape( numpy.repeat( ml_image[:, :, numpy.newaxis], 3, axis=2 ), newshape=( ml_image.shape[0], ml_image.shape[1], 3 ) )

    # export image #
    image.imsave( ml_path, ml_image )

def ml_data_image_concat( ml_image_a, ml_image_b ):

    # return concatenated images #
    return numpy.c_[ml_image_a, ml_image_b ]

def ml_data_image_decimate_grid( ml_data, ml_grid ):

    # parsing data-set #
    for ml_parse in range( ml_data.shape[0] ):

        # parsing pixels #
        for ml_x in range( ml_data.shape[1] ):

            # parsing pixels #
            for ml_y in range( ml_data.shape[2] ):

                # check decimation condition #
                if ( ( ml_x % ml_grid ) != 0 ) or ( ( ml_y % ml_grid ) != 0 ):

                    ml_data[ml_parse][ml_x][ml_y][0] = 0
                    ml_data[ml_parse][ml_x][ml_y][1] = 0
                    ml_data[ml_parse][ml_x][ml_y][2] = 0

##
##  script - vector manipulation
##

def ml_data_vector_save( ml_vector, ml_path ):

    # export vector #
    numpy.savetxt( ml_path, ml_vector )

def ml_data_vector_load( ml_path ):

    # read and return vector #
    return numpy.loadtxt( ml_path )

