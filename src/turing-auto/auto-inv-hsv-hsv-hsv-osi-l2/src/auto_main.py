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
ml_apar = argparse.ArgumentParser( description='Research auto-encoder built with tensorflow' )

# argument directive #
ml_apar.add_argument( '-m', '--mode'   , type=str, help='script mode'  )
ml_apar.add_argument( '-e', '--epoch'  , type=int, help='epoch length' )
ml_apar.add_argument( '-b', '--batch'  , type=int, help='batch length' )
ml_apar.add_argument( '-i', '--input'  , type=str, help='input path'   )
ml_apar.add_argument( '-o', '--output' , type=str, help='output path'  )
ml_apar.add_argument( '-w', '--width'  , type=int, help='image width'  )
ml_apar.add_argument( '-1', '--layer1' , type=int, help='layer size'   )
ml_apar.add_argument( '-2', '--layer2' , type=int, help='layer size'   )
ml_apar.add_argument( '-n', '--network', type=str, help='network path' )
ml_apar.add_argument( '-u', '--start'  , type=int, help='range start'  )
ml_apar.add_argument( '-v', '--stop'   , type=int, help='range stop'   )

# parse argument #
ml_args = ml_apar.parse_args()

##
##  script - network hyper-parameter
##

# network hyper-parameter #
ml_h_input   = ml_args.width ** 2
ml_h_hidden1 = ml_args.layer1
ml_h_hidden2 = ml_args.layer2

##
##   script - network parameter
##

# network parameter : weights #
ml_p_w1 = tf.Variable( tf.random_normal( [ ml_h_input  , ml_h_hidden1 ], stddev=0.05 ) )
ml_p_w2 = tf.Variable( tf.random_normal( [ ml_h_hidden1, ml_h_hidden2 ], stddev=0.05 ) )
ml_p_w3 = tf.Variable( tf.random_normal( [ ml_h_hidden2, ml_h_hidden1 ], stddev=0.05 ) )
ml_p_w4 = tf.Variable( tf.random_normal( [ ml_h_hidden1, ml_h_input   ], stddev=0.05 ) )

# network parameter : biases #
ml_p_b1 = tf.Variable( tf.zeros( [ ml_h_hidden1 ] ) )
ml_p_b2 = tf.Variable( tf.zeros( [ ml_h_hidden2 ] ) )
ml_p_b3 = tf.Variable( tf.zeros( [ ml_h_hidden1 ] ) )
ml_p_b4 = tf.Variable( tf.zeros( [ ml_h_input   ] ) )

##
##   script - network topology
##

# network topology : input layer #
ml_g_input = tf.placeholder( tf.float32, [ None, ml_h_input ] )

# network topology : hidden layer #
ml_g_hidden_a = tf.nn.sigmoid( tf.add( tf.matmul( ml_g_input, ml_p_w1 ), ml_p_b1 ) )

# network topology : hidden layer #
ml_g_hidden_b = tf.nn.sigmoid( tf.add( tf.matmul( ml_g_hidden_a, ml_p_w2 ), ml_p_b2 ) )

# network topology : hidden layer #
ml_g_hidden_c = tf.nn.sigmoid( tf.add( tf.matmul( ml_g_hidden_b, ml_p_w3 ), ml_p_b3 ) )

# network topology : output layer #
ml_g_output = tf.nn.sigmoid(  tf.add( tf.matmul( ml_g_hidden_c, ml_p_w4 ), ml_p_b4 ) )

##
##   script - network sub-topology
##

# network topology : input layer #
ml_s1_input = tf.placeholder( tf.float32, [ None, ml_h_input ] )

# network topology : output layer #
ml_s1_hidden = tf.nn.sigmoid( tf.add( tf.matmul( ml_s1_input, ml_p_w1 ),ml_p_b1 ) )

# network topology : output layer #
ml_s1_output = tf.nn.sigmoid( tf.add( tf.matmul( ml_s1_hidden, ml_p_w2 ),ml_p_b2 ) )

# network topology : input layer #
ml_s2_input = tf.placeholder( tf.float32, [ None, ml_h_hidden2 ] )

# network topology : output layer #
ml_s2_hidden = tf.nn.sigmoid( tf.add( tf.matmul( ml_s2_input, ml_p_w3 ), ml_p_b3 ) )

# network topology : output layer #
ml_s2_output = tf.nn.sigmoid( tf.add( tf.matmul( ml_s2_hidden, ml_p_w4 ), ml_p_b4 ) )

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
if ( ( ml_args.mode == 'train' ) or ( ml_args.mode == 'retrain' ) ):

    # import data #
    ml_data = td.ml_data_import( ml_args.input, ml_args.width )

    # format data #
    ml_data = td.ml_data_format_y( ml_data )

    # reshape data #
    ml_data = ml_data.reshape( -1, ml_h_input )

    # training and validation loss data #
    ml_data, ml_train, ml_valid = td.ml_data_split( ml_data, 0.8, ml_args.batch )

    # minibatch count #
    ml_count = td.ml_data_batch_count( ml_data, ml_args.batch )

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

        # randomise data #
        ml_data = td.ml_data_shuffle( ml_data, td.ml_data_random( ml_data ) )

        # network training : optimisation steps #
        for ml_stocastic in range( ml_count ):

            # extract minibatch #
            ml_batch = td.ml_data_batch( ml_data, ml_args.batch, ml_stocastic )

            # optimisation step #
            ml_session.run( ml_o_mopt, feed_dict={ ml_g_input : ml_batch } )

        # compute training loss #
        ml_t_loss = ml_session.run( [ ml_o_loss ], feed_dict={ ml_g_input : ml_train } )

        # compute validation loss #
        ml_v_loss = ml_session.run( [ ml_o_loss ], feed_dict={ ml_g_input : ml_valid } )

        # append to loss vectors #
        ml_loss.append( [ ml_t_loss[0], ml_v_loss[0] ] )

        # display information on loss #
        print( 'epoch :', "{:06d}".format(ml_epoch), ' : t_loss =', "{:0.4e}".format(ml_t_loss[0]), ' : v_loss =', "{:0.4e}".format(ml_v_loss[0]) )

    # export loss #
    td.ml_data_vector_save( ml_loss, ml_args.network + '/loss-data' )

    # export network #
    ml_network.save( ml_session, ml_args.network )

# script mode #
elif ( ml_args.mode == 'auto' ):

    # import data #
    ml_data = td.ml_data_import( ml_args.input, ml_args.width )

    # format data #
    ml_data = td.ml_data_format_y( ml_data )

    # reshape data #
    ml_data = ml_data.reshape( -1, ml_h_input )

    # extract data range #
    ml_data = td.ml_data_range( ml_data, ml_args.start, ml_args.stop )

    # tensorflow session #
    ml_session = tf.Session()

    # import network #
    ml_network.restore( ml_session, ml_args.network )

    # compute auto-encoded #
    ml_auto = ml_session.run( ml_g_output, feed_dict={ ml_g_input : ml_data } )

    # reshape range #
    ml_data = ml_data.reshape( -1, ml_args.width, ml_args.width )
    ml_auto = ml_auto.reshape( -1, ml_args.width, ml_args.width )

    # parsing range #
    for ml_export in range( ml_data.shape[0] ):

        # export auto-encoded with prior #
        td.ml_data_image_save( td.ml_data_image_concat( ml_data[ml_export], ml_auto[ml_export] ), ml_args.output + '/image-{:06d}.png'.format( ml_export + ml_args.start ) )

# script mode #
elif ( ml_args.mode == 'encode' ):

    # import data #
    ml_data = td.ml_data_import( ml_args.input, ml_args.width )

    # format data #
    ml_data = td.ml_data_format_y( ml_data )

    # reshape data #
    ml_data = ml_data.reshape( -1, ml_h_input )

    # extract data range #
    ml_data = td.ml_data_range( ml_data, ml_args.start, ml_args.stop )

    # tensorflow session #
    ml_session = tf.Session()

    # import network #
    ml_network.restore( ml_session, ml_args.network )

    # compute projection - encode #
    ml_encode = ml_session.run( ml_s1_output, feed_dict={ ml_s1_input  : ml_data } )

    # export projection #
    td.ml_data_vector_save( ml_encode, ml_args.output + '/vector-{:06d}'.format( ml_args.start ) )

# script mode #
elif ( ml_args.mode == 'decode' ):

    # import vector #
    ml_data = td.ml_data_vector_load( ml_args.input )

    # check consistency #
    if ( ml_data.shape[1] != ml_h_hidden2 ):

        # send message #
        sys.exit( 'turing : error : inconsistent vector' )

    # tensorflow session #
    ml_session = tf.Session()

    # import network #
    ml_network.restore( ml_session, ml_args.network )

    # compute deprojection - decode #
    ml_decode = ml_session.run( ml_s2_output, feed_dict={ ml_s2_input : ml_data } )

    # reshape range #
    ml_decode = ml_decode.reshape( -1, ml_args.width, ml_args.width )

    # parsing range #
    for ml_parse in range( ml_data.shape[0] ):

        # export decoded #
        td.ml_data_image_save( ml_decode[ml_parse], ml_args.output + '/image-{:06d}.png'.format( ml_parse ) )

