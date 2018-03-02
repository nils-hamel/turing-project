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
import sys
import os
import auto_data as td
import tensorflow as tf

##
##   script - argument and parameter
##

# create argument parser #
ml_apar = argparse.ArgumentParser( description='Research auto-encoder built using tensorflow' )

# argument directive #
ml_apar.add_argument( '-m', '--mode'   , type=str, help='script mode'  )
ml_apar.add_argument( '-e', '--epoch'  , type=int, help='epoch length' )
ml_apar.add_argument( '-b', '--batch'  , type=int, help='batch length' )
ml_apar.add_argument( '-i', '--input'  , type=str, help='input path'   )
ml_apar.add_argument( '-d', '--inputl' , type=str, help='input path'   )
ml_apar.add_argument( '-D', '--inputh' , type=str, help='input path'   )
ml_apar.add_argument( '-o', '--output' , type=str, help='output path'  )
ml_apar.add_argument( '-w', '--widthl' , type=int, help='image width'  )
ml_apar.add_argument( '-W', '--widthh' , type=int, help='image width'  )
ml_apar.add_argument( '-1', '--layer1' , type=int, help='layer size'   )
ml_apar.add_argument( '-n', '--network', type=str, help='network path' )
ml_apar.add_argument( '-u', '--start'  , type=int, help='range start'  )
ml_apar.add_argument( '-v', '--stop'   , type=int, help='range stop'   )

# parse argument #
ml_args = ml_apar.parse_args()

##
##  script - network hyper-parameter
##

# network hyper-parameter #
ml_h_width_l = ml_args.widthl
ml_h_width_h = ml_args.widthh
ml_h_flat_l  = ml_args.widthl * ml_args.widthl * ml_args.widthl
ml_h_flat_h  = ml_args.widthh * ml_args.widthh * ml_args.widthh
ml_h_hidden  = ml_args.layer1

##
##   script - network parameter
##

# network parameter : weights #
ml_p_w1 = tf.Variable( tf.random_normal( [ ml_h_flat_l, ml_h_hidden ], stddev = 0.05 ) )
ml_p_w2 = tf.Variable( tf.random_normal( [ ml_h_hidden, ml_h_flat_h ], stddev = 0.05 ) )

# network parameter : biases #
ml_p_b1 = tf.Variable( tf.zeros( [ ml_h_hidden ] ) )
ml_p_b2 = tf.Variable( tf.zeros( [ ml_h_flat_h ] ) )

##
##   script - network topology
##

# network topology : input layer #
ml_g_input = tf.placeholder( tf.float32, [ None, ml_h_width_l, ml_h_width_l, ml_h_width_l ] )

# network topology : input layer #
ml_g_input_flat = tf.reshape( ml_g_input, [ -1, ml_h_flat_l ] )

# network topology : hidden layer #
ml_g_hidden = tf.nn.sigmoid( tf.add( tf.matmul( ml_g_input_flat, ml_p_w1 ), ml_p_b1 ) )

# network topology : output layer #
ml_g_output_flat = tf.nn.sigmoid( tf.add( tf.matmul( ml_g_hidden, ml_p_w2 ), ml_p_b2 ) )

# network topology : output layer #
ml_g_output = tf.reshape( ml_g_output_flat, [ -1, ml_h_width_h, ml_h_width_h, ml_h_width_h ] )

# network topology : target layer #
ml_g_target = tf.placeholder( tf.float32, [ None, ml_h_width_h, ml_h_width_h, ml_h_width_h ] )

##
##   script - network objective function
##

# objective function : mean(L2) #
ml_o_loss = tf.reduce_mean( tf.pow( tf.subtract( ml_g_output, ml_g_target ), 2 ) )

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
if ( ( ml_args.mode == 'train' ) or ( ml_args.mode == 'retrain' ) ):

    # import data #
    ml_data_l = td.ml_data_import( ml_args.inputl, ml_args.widthl )
    ml_data_h = td.ml_data_import( ml_args.inputh, ml_args.widthh )

    # data format #
    ml_data_l = td.ml_data_format_float( ml_data_l )
    ml_data_h = td.ml_data_format_float( ml_data_h )

    # training and validation loss data #
    ml_data_l, ml_train_l, ml_valid_l = td.ml_data_split( ml_data_l, 0.8, ml_args.batch )
    ml_data_h, ml_train_h, ml_valid_h = td.ml_data_split( ml_data_h, 0.8, ml_args.batch )

    # minibatch count #
    ml_count = td.ml_data_batch_count( ml_data_l, ml_args.batch )

    # tensorflow session #
    ml_session = tf.Session()

    # tensorflow variable #
    ml_session.run( ml_variables )

    # script mode #
    if ( ml_args.mode == 'retrain' ):

        # import network #
        ml_network.restore( ml_session, ml_args.network )

    # initialise loss history #
    ml_loss = []

    # network training : epochs #
    for ml_epoch in range( ml_args.epoch ):

        # create random index #
        ml_index = td.ml_data_random( ml_data_l )

        # randomise data #
        ml_data_l = td.ml_data_shuffle( ml_data_l, ml_index )
        ml_data_h = td.ml_data_shuffle( ml_data_h, ml_index )

        # network training : optimisation steps #
        for ml_stocastic in range( ml_count ):

            # extract minibatch #
            ml_batch_l = td.ml_data_batch( ml_data_l, ml_args.batch, ml_stocastic )
            ml_batch_h = td.ml_data_batch( ml_data_h, ml_args.batch, ml_stocastic )

            # optimisation step #
            ml_session.run( ml_o_mopt, feed_dict={ ml_g_input : ml_batch_l, ml_g_target : ml_batch_h } )

        # compute training loss #
        ml_t_loss = ml_session.run( [ ml_o_loss ], feed_dict={ ml_g_input : ml_train_l, ml_g_target : ml_train_h } )

        # compute validation loss #
        ml_v_loss = ml_session.run( [ ml_o_loss ], feed_dict={ ml_g_input : ml_valid_l, ml_g_target : ml_valid_h } )

        # append to loss vectors #
        ml_loss.append( [ ml_t_loss[0], ml_v_loss[0] ] )

        # display information on loss #
        print( 'epoch :', "{:06d}".format(ml_epoch), ': t_loss =', "{:0.4e}".format(ml_t_loss[0]), ': v_loss =', "{:0.4e}".format(ml_v_loss[0]) )

    # export loss #
    td.ml_data_vector_save( ml_loss, ml_args.network + '/loss-data' )

    # export network #
    ml_network.save( ml_session, ml_args.network )

# script mode #
elif ( ml_args.mode == 'tran' ):

    # import data #
    ml_data_l = td.ml_data_import( ml_args.inputl, ml_args.widthl )

    # data format #
    ml_data_l = td.ml_data_format_float( ml_data_l )

    # extract data range #
    ml_data_l = td.ml_data_range( ml_data_l, ml_args.start, ml_args.stop )

    # tensorflow session #
    ml_session = tf.Session()

    # import network #
    ml_network.restore( ml_session, ml_args.network )

    # compute auto-encoded #
    ml_data_h = ml_session.run( ml_g_output, feed_dict={ ml_g_input : ml_data_l } )

    # convert auto-encoded #
    ml_data_h = td.ml_data_format_uint8( ml_data_h )

    # parsing data range #
    for ml_export in range( ml_data_l.shape[0] ):

        # export original raster #
        td.ml_data_raster_save( ml_data_l[ml_export], ml_args.output + '/raster-{:06d}-orig.ras'.format( ml_export + ml_args.start ) )

        # export auto-encoded raster #
        td.ml_data_raster_save( ml_data_h[ml_export], ml_args.output + '/raster-{:06d}-tran.ras'.format( ml_export + ml_args.start ) )

