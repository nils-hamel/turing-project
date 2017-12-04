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
import sys
import os
import auto_data
import tensorflow as tf
import matplotlib.pyplot as plt

##
##   script - argument and parameter
##

# create argument parser #
ml_apar = argparse.ArgumentParser( description='Research auto-encoder built with tensorflow' )

# argument directive #
ml_apar.add_argument( '-d', '--input'  , type=str, help='import path'  )
ml_apar.add_argument( '-s', '--size'   , type=int, help='image size'   )
ml_apar.add_argument( '-1', '--hidden' , type=int, help='hidden size'  )
ml_apar.add_argument( '-m', '--mode'   , type=str, help='script mode'  )
ml_apar.add_argument( '-e', '--epoch'  , type=int, help='epoch number' )
ml_apar.add_argument( '-b', '--batch'  , type=int, help='batch size'   )
ml_apar.add_argument( '-n', '--network', type=str, help='network path' )
ml_apar.add_argument( '-u', '--start'  , type=int, help='range start'  )
ml_apar.add_argument( '-v', '--stop'   , type=int, help='range stop'   )
ml_apar.add_argument( '-x', '--output' , type=str, help='export path'  )

# parse argument #
ml_args = ml_apar.parse_args()

##
##  script - network hyper-parameter
##

# network hyper-parameter #
ml_h_input  = ml_args.size ** 2
ml_h_hidden = ml_args.hidden

##
##   script - network parameter
##

# network parameter : weights #
ml_p_w1 = tf.Variable( tf.random_normal( [ ml_h_input, ml_h_hidden ] ) )
ml_p_w2 = tf.Variable( tf.random_normal( [ ml_h_hidden, ml_h_input ] ) )

# network parameter : biases #
ml_p_b1 = tf.Variable( tf.random_normal( [ ml_h_hidden ] ) )
ml_p_b2 = tf.Variable( tf.random_normal( [ ml_h_input  ] ) )

##
##   script - network topology
##

# network topology : input layer #
ml_g_input = tf.placeholder( tf.float32, [ None, ml_h_input ] )

# network topology : hidden layer #
ml_g_hidden = tf.nn.sigmoid( tf.add( tf.matmul( ml_g_input, ml_p_w1 ), ml_p_b1 ) )

# network topology : output layer #
ml_g_output = tf.nn.sigmoid( tf.add( tf.matmul( ml_g_hidden, ml_p_w2 ), ml_p_b2 ) )

##
##   script - network sub-topology
##

# network topology : input layer #
ml_s1_input = tf.placeholder( tf.float32, [ None, ml_h_input ] )

# network topology : output layer #
ml_s1_output = tf.nn.sigmoid( tf.add( tf.matmul( ml_s1_input, ml_p_w1 ),ml_p_b1 ) )

# network topology : input layer #
ml_s2_input = tf.placeholder( tf.float32, [ None, ml_h_hidden ] )

# network topology : output layer #
ml_s2_output = tf.nn.sigmoid( tf.add( tf.matmul( ml_s2_input, ml_p_w2 ), ml_p_b2 ) )

##
##   script - network objective function
##

# objective function : mean(L2) #
ml_o_loss = tf.reduce_mean( tf.pow( tf.subtract( ml_g_output, ml_g_input ), 2 ) )

##
##   script - network optimisation
##

# optimisation algorithm #
ml_o_mopt = tf.train.AdamOptimizer( 0.001 ).minimize( ml_o_loss )

##
##   script - main function
##

# variable initialisation #
ml_variables = tf.global_variables_initializer()

# network i/o management #
ml_network = tf.train.Saver()

# script mode #
if ( ml_args.mode == 'train' ):

    # import data #
    ml_data = auto_data.ml_data_import( ml_args.input, ml_h_input )

    # training and validation data #
    ml_data, ml_valid = auto_data.ml_data_split( ml_data, 0.8 )

    # minibatch count #
    ml_count = auto_data.ml_data_batch_count( ml_data, ml_args.batch )

    # tensorflow session #
    ml_session = tf.Session()

    # tensorflow variable #
    ml_session.run( ml_variables )

    # network training : epochs #
    for ml_epoch in range( ml_args.epoch ):

        # randomise data #
        ml_data = auto_data.ml_data_shuffle( ml_data, auto_data.ml_data_random( ml_data ) )

        # network training : optimisation steps #
        for ml_stocastic in range( ml_count ):

            # extract minibatch #
            ml_batch = auto_data.ml_data_batch( ml_data, ml_args.batch, ml_stocastic )

            # optimisation step #
            _, ml_t_loss = ml_session.run( [ ml_o_mopt, ml_o_loss ], feed_dict={ ml_g_input : ml_batch } )

        # compute validation loss #
        ml_v_loss = ml_session.run( [ ml_o_loss ], feed_dict={ ml_g_input : ml_valid } )

        # display information on loss #
        print( 'epoch :', "{:06d}".format(ml_epoch), ' : t_loss =', "{:0.4e}".format(ml_t_loss), ' : v_loss =', "{:0.4e}".format(ml_v_loss[0]) )

    # export network #
    ml_network.save( ml_session, ml_args.network )

# script mode #
elif ( ml_args.mode == 'auto' ):

    # import data #
    ml_data = auto_data.ml_data_import( ml_args.input, ml_h_input )

    # check consistency #
    if ( auto_data.ml_data_range( ml_data, ml_args.start, ml_args.stop ) == False ):

        # send message #
        sys.exit( 'turing : error : range specification' )

    # extract data range #
    ml_data = ml_data[ml_args.start:ml_args.stop]

    # tensorflow session #
    ml_session = tf.Session()

    # tensorflow variables #
    ml_session.run( ml_variables )

    # import network #
    ml_network.restore( ml_session, ml_args.network )

    # compute auto-encoded #
    ml_auto = ml_session.run( ml_g_output, feed_dict={ ml_g_input : ml_data } )

    # reshape range #
    ml_data = ml_data.reshape( -1, ml_args.size, ml_args.size )
    ml_auto = ml_auto.reshape( -1, ml_args.size, ml_args.size )

    # parsing range #
    for ml_export in range( ml_data.shape[0] ):

        # export auto-encoded with prior #
        auto_data.ml_data_image_save( auto_data.ml_data_image_concat( ml_data[ml_export], ml_auto[ml_export] ), ml_args.output + '/image-{:06d}.png'.format( ml_export + ml_args.start ) )

# script mode #
elif ( ml_args.mode == 'encode' ):

    # import data #
    ml_data = auto_data.ml_data_import( ml_args.input, ml_h_input )

    # check consistency #
    if ( auto_data.ml_data_range( ml_data, ml_args.start, ml_args.stop ) == False ):

        # send message #
        sys.exit( 'turing : error : range specification' )

    # extract data range #
    ml_data = ml_data[ml_args.start:ml_args.stop]

    # tensorflow session #
    ml_session = tf.Session()

    # tensorflow variables #
    ml_session.run( ml_variables )

    # import network #
    ml_network.restore( ml_session, ml_args.network )

    # compute projection - encode #
    ml_encode = ml_session.run( ml_s1_output, feed_dict={ ml_s1_input  : ml_data } )

    # export projection #
    auto_data.ml_data_vector_save( ml_encode, ml_args.output + '/vector-{:06d}'.format( ml_args.start ) )

# script mode #
elif ( ml_args.mode == 'decode' ):

    # import vector #
    ml_data = auto_data.ml_data_vector_load( ml_args.input )

    # check consistency #
    if ( ml_data.shape[1] != ml_h_hidden ):

        # send message #
        sys.exit( 'turing : error : inconsistent vector' )

    # tensorflow session #
    ml_session = tf.Session()

    # tensorflow variables #
    ml_session.run( ml_variables )

    # import network #
    ml_network.restore( ml_session, ml_args.network )

    # compute deprojection - decode #
    ml_decode = ml_session.run( ml_s2_output, feed_dict={ ml_s2_input : ml_data } )

    # reshape range #
    ml_decode = ml_decode.reshape( -1, ml_args.size, ml_args.size )

    # parsing range #
    for ml_parse in range( ml_data.shape[0] ):

        # export decoded #
        auto_data.ml_data_image_save( ml_decode[ml_parse], ml_args.output + '/image-{:06d}.png'.format( ml_parse ) )

# script mode #
elif ( ml_args.mode == 'view' ):

    # tensorflow session #
    ml_session = tf.Session()

    # tensorflow variables #
    ml_session.run( ml_variables )

    # import network #
    ml_network.restore( ml_session, ml_args.network )

    # export weights #
    auto_data.ml_data_vector_save( ml_p_w1.eval( ml_session ), ml_args.output + '/w-layer-ih' )
    auto_data.ml_data_vector_save( ml_p_w2.eval( ml_session ), ml_args.output + '/w-layer-ho' )

    # export biases #
    auto_data.ml_data_vector_save( ml_p_b1.eval( ml_session ), ml_args.output + '/b-layer-ih' )
    auto_data.ml_data_vector_save( ml_p_b2.eval( ml_session ), ml_args.output + '/b-layer-ho' )

