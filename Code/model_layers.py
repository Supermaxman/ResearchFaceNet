# MIT License
# 
# Copyright (c) 2018 Maxwell Weinzierl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


def conv_layer(inputs, filters, kernel_size, strides, padding, name, 
               batch_norm=False,
               is_training=None,
               kernel_initializer=layers.xavier_initializer_conv2d(),
               kernel_regularizer=None,
               activation=tf.nn.relu,
               skip_connection=False):

    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding,
                                activation=None, kernel_initializer=kernel_initializer, 
                                kernel_regularizer=kernel_regularizer,
                                name='conv')

        if batch_norm:
            conv = tf.layers.batch_normalization(conv, training=is_training, momentum=0.995, epsilon=0.001, name='bn')

        if activation != None:
            conv = activation(conv, name='activ')

        if skip_connection:
            conv = conv + inputs

    return conv


def block35(name, inputs, branch_conv, up_conv, out_activ_fn, scale=1.0):
    with tf.variable_scope(name):
        with tf.variable_scope('branch_0'):
            tower_conv = branch_conv(inputs, 32, 1, strides=1, padding='SAME', name='conv_1x1')
            
        with tf.variable_scope('branch_1'):
            tower_conv1_0 = branch_conv(inputs, 32, 1, strides=1, padding='SAME', name='conv_0a_1x1')
            tower_conv1_1 = branch_conv(tower_conv1_0, 32, 3, strides=1, padding='SAME', name='conv_0b_3x3')
        with tf.variable_scope('branch_2'):
            tower_conv2_0 = branch_conv(inputs, 32, 1, strides=1, padding='SAME', name='conv_0a_1x1')
            tower_conv2_1 = branch_conv(tower_conv2_0, 32, 3, strides=1, padding='SAME', name='conv_0b_3x3')
            tower_conv2_2 = branch_conv(tower_conv2_1, 32, 3, strides=1, padding='SAME', name='conv_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)

        up = up_conv(mixed, inputs.get_shape()[3], 1, strides=1, padding='SAME', name='conv_1x1')

        out = inputs + (scale * up)
        if out_activ_fn:
            out = out_activ_fn(out)
    return out
        

def block17(name, inputs, branch_conv, up_conv, out_activ_fn, scale=1.0):
    with tf.variable_scope(name):
        with tf.variable_scope('branch_0'):
            tower_conv = branch_conv(inputs, 128, 1, strides=1, padding='SAME', name='conv_1x1')
        with tf.variable_scope('branch_1'):
            tower_conv1_0 = branch_conv(inputs, 128, 1, strides=1, padding='SAME', name='conv_0a_1x1')
            tower_conv1_1 = branch_conv(tower_conv1_0, 128, [1, 7], strides=1, padding='SAME', name='conv_0b_1x7')
            tower_conv1_2 = branch_conv(tower_conv1_1, 128, [7, 1], strides=1, padding='SAME', name='conv_0c_7x1')

        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = up_conv(mixed, inputs.get_shape()[3], 1, strides=1, padding='SAME', name='conv_1x1')

        out = inputs + (scale * up)
        if out_activ_fn:
            out = out_activ_fn(out)
    return out

def block8(name, inputs, branch_conv, up_conv, out_activ_fn, scale=1.0):
    with tf.variable_scope(name):
        with tf.variable_scope('branch_0'):
            tower_conv = branch_conv(inputs, 192, 1, strides=1, padding='SAME', name='conv_1x1')
        with tf.variable_scope('branch_1'):
            tower_conv1_0 = branch_conv(inputs, 192, 1, strides=1, padding='SAME', name='conv_0a_1x1')
            tower_conv1_1 = branch_conv(tower_conv1_0, 192, [1, 3], strides=1, padding='SAME', name='conv_0b_1x3')
            tower_conv1_2 = branch_conv(tower_conv1_1, 192, [3, 1], strides=1, padding='SAME', name='conv_0c_3x1')
            
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)

        up = up_conv(mixed, inputs.get_shape()[3], 1, strides=1, padding='SAME', name='conv_1x1')

        out = inputs + (scale * up)
        if out_activ_fn:
            out = out_activ_fn(out)
    return out

def reduction_a(name, inputs, k, l, m, n, conv):
    with tf.variable_scope(name):
        with tf.variable_scope('branch_0'):
            tower_conv = conv(inputs, n, 3, strides=2, padding='VALID', name='conv_1a_3x3')
        with tf.variable_scope('branch_1'):
            tower_conv1_0 = conv(inputs, k, 1, strides=1, padding='SAME', name='conv_0a_1x1')
            tower_conv1_1 = conv(tower_conv1_0, l, 3, strides=1, padding='SAME', name='conv_0b_3x3')
            tower_conv1_2 = conv(tower_conv1_1, m, 3, strides=2, padding='VALID', name='conv_1a_3x3')
        with tf.variable_scope('branch_2'):
            tower_pool = tf.layers.max_pooling2d(inputs, 3, strides=2, padding='VALID', name='maxpool_1a_3x3')
        out = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return out

    
def reduction_b(name, inputs, conv):
    with tf.variable_scope(name):
        with tf.variable_scope('branch_0'):
            tower_conv = conv(inputs, 256, 1, strides=1, padding='SAME', name='conv_0a_1x1')
            tower_conv_1 = conv(tower_conv, 384, 3, strides=2, padding='VALID', name='conv_1a_3x3')
        with tf.variable_scope('branch_1'):
            tower_conv1 = conv(inputs, 256, 1, strides=1, padding='SAME', name='conv_0a_1x1')
            tower_conv1_1 = conv(tower_conv1, 256, 3, strides=2, padding='VALID', name='conv_1a_3x3')
        with tf.variable_scope('branch_2'):
            tower_conv2 = conv(inputs, 256, 1, strides=1, padding='SAME', name='conv_0a_1x1')
            tower_conv2_1 = conv(tower_conv2, 256, 3, strides=1, padding='SAME', name='conv_0b_3x3')
            tower_conv2_2 = conv(tower_conv2_1, 256, 3, strides=2, padding='VALID', name='conv_1a_3x3')
        with tf.variable_scope('branch_3'):
            tower_pool = tf.layers.max_pooling2d(inputs, 3, strides=2, padding='VALID', name='maxpool_1a_3x3')

        out = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)
    return out


def inception_resnet_v1_stem(name, inputs, conv):
    with tf.variable_scope(name):
        # 149 x 149 x 32
        conv_1a_3x3 = conv(inputs, 32, 3, strides=2, padding='VALID', name='conv_1a_3x3')

        # 147 x 147 x 32
        conv_2a_3x3 = conv(conv_1a_3x3, 32, 3, strides=1, padding='VALID', name='conv_2a_3x3')

        # 147 x 147 x 64
        conv_2b_3x3 = conv(conv_2a_3x3, 64, 3, strides=1, padding='SAME', name='conv_2b_3x3')

        # 73 x 73 x 64
        maxpool_3a_3x3 = tf.layers.max_pooling2d(conv_2b_3x3, 3, strides=2, padding='VALID', name='maxpool_3a_3x3')

        # 73 x 73 x 80
        conv_3b_1x1 = conv(maxpool_3a_3x3, 80, 1, strides=1, padding='VALID', name='conv_3b_1x1')

        # 71 x 71 x 192
        conv_4a_3x3 = conv(conv_3b_1x1, 192, 3, strides=1, padding='VALID', name='conv_4a_3x3')

        # 35 x 35 x 256
        conv_4b_3x3 = conv(conv_4a_3x3, 256, 3, strides=2, padding='VALID', name='conv_4b_3x3')
        return conv_4b_3x3


def conv_repeat(name, inputs, repeat_count, repeat_layer):

    with tf.variable_scope(name):
        repeat_net = inputs
        for conv_idx in range(repeat_count):
            repeat_net = repeat_layer(name='repeat_{}'.format(conv_idx), inputs=repeat_net)

    return repeat_net


def l2_pool(input, pool_size=3, stride=2, padding='SAME', name='l2_pool'):
    return tf.sqrt(tf.nn.avg_pool(tf.square(input), [1, pool_size, pool_size, 1], [1, stride, stride, 1], padding, name=name))

def inception_stem(layer_name,
                   prev_layer, is_training,
                   conv7x7_size,
                   conv3x3_1x1_size,
                   conv3x3_size,
                   kernel_initializer=layers.xavier_initializer_conv2d(),
                   kernel_regularizer=None,
                   activation_fn=tf.nn.relu,
                   batch_norm=True):
    with tf.variable_scope(layer_name):
        conv7x7 = conv_layer(prev_layer, conv7x7_size, [7, 7], 2, 'SAME', 
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=kernel_regularizer, 
                             batch_norm=batch_norm, is_training=is_training,
                             activation=activation_fn, name='conv7x7')

        pool7x7 = tf.layers.max_pooling2d(conv7x7, [3, 3], 2, 'SAME', name='pool7x7')
        
        lrn7x7 = tf.nn.local_response_normalization(pool7x7, name='lrn7x7')

        conv3x3_1x1 = conv_layer(lrn7x7, conv3x3_1x1_size, [1, 1], 1, 'SAME', 
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer, 
                                 batch_norm=batch_norm, is_training=is_training,
                                 activation=activation_fn, name='conv3x3_1x1')

        conv3x3 = conv_layer(conv3x3_1x1, conv3x3_size, [3, 3], 1, 'SAME', 
                             kernel_initializer=kernel_initializer, 
                             kernel_regularizer=kernel_regularizer,
                             batch_norm=batch_norm, is_training=is_training,
                             activation=activation_fn, name='conv3x3')
        
        lrn3x3 = tf.nn.local_response_normalization(conv3x3, name='lrn3x3')

        pool3x3 = tf.layers.max_pooling2d(lrn3x3, [3, 3], 2, 'SAME', name='pool3x3')

        return pool3x3


def inception_module(layer_name, 
                     prev_layer, is_training, 
                     conv1x1_size, 
                     conv3x3_1x1_size, conv3x3_size, 
                     conv5x5_1x1_size, conv5x5_size, 
                     pool3x3_1x1_size=0,
                     conv3x3_stride=1,
                     conv5x5_stride=1,
                     pool_type='MAX',
                     pool_stride=1,
                     kernel_initializer=layers.xavier_initializer_conv2d(),
                     kernel_regularizer=None,
                     activation_fn=tf.nn.relu,
                     batch_norm=True):

    with tf.variable_scope(layer_name):
        combined_list = []
        if conv1x1_size > 0:
            with tf.variable_scope('conv1x1'):
                conv1x1 = conv_layer(prev_layer, conv1x1_size, [1, 1], 1, 'SAME', 
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=kernel_regularizer, 
                                     batch_norm=batch_norm, is_training=is_training,
                                     activation=activation_fn, name='conv1x1')
                
                combined_list.append(conv1x1)
        if conv3x3_size > 0:
            with tf.variable_scope('conv3x3'):
                conv3x3_1x1 = conv_layer(prev_layer, conv3x3_1x1_size, [1, 1], 1, 'SAME', 
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer, 
                                         batch_norm=batch_norm, is_training=is_training,
                                         activation=activation_fn, name='conv3x3_1x1')
                conv3x3 = conv_layer(conv3x3_1x1, conv3x3_size, [3, 3], conv3x3_stride, 'SAME', 
                                     kernel_initializer=kernel_initializer, 
                                     kernel_regularizer=kernel_regularizer,
                                     batch_norm=batch_norm, is_training=is_training,
                                     activation=activation_fn, name='conv3x3')
                combined_list.append(conv3x3)

        if conv5x5_size > 0:
            with tf.variable_scope('conv5x5'):
                conv5x5_1x1 = conv_layer(prev_layer, conv5x5_1x1_size, [1, 1], 1, 'SAME', 
                                         kernel_initializer=kernel_initializer, 
                                         kernel_regularizer=kernel_regularizer,
                                         batch_norm=batch_norm, is_training=is_training,
                                         activation=activation_fn, name='conv5x5_1x1')
                conv5x5 = conv_layer(conv5x5_1x1, conv5x5_size, [5, 5], conv5x5_stride, 'SAME', 
                                     kernel_initializer=kernel_initializer, 
                                     kernel_regularizer=kernel_regularizer,
                                     batch_norm=batch_norm, is_training=is_training,
                                     activation=activation_fn, name='conv5x5')
                combined_list.append(conv5x5)

        with tf.variable_scope('pool3x3'):

            if pool_type == 'MAX':
                pool3x3 = tf.layers.max_pooling2d(prev_layer, [3, 3], pool_stride, 'SAME', name='max_pool3x3')
            elif pool_type == 'L2':
                pool3x3 = l2_pool(prev_layer, 3, pool_stride, 'SAME', name='l2_pool3x3')
            else:
                raise ValueError('Unknown pooling type.')

            if pool3x3_1x1_size > 0:
                pool3x3_1x1 = conv_layer(pool3x3, pool3x3_1x1_size, [1, 1], 1, 'SAME',
                                         kernel_initializer=kernel_initializer, 
                                         kernel_regularizer=kernel_regularizer,
                                         batch_norm=batch_norm, is_training=is_training,
                                         activation=activation_fn, name='pool3x3_1x1')
            else:
                pool3x3_1x1 = pool3x3

            combined_list.append(pool3x3_1x1)

        conv_combined = tf.concat(combined_list, 3)

        return conv_combined

#K-NN impl for embedding lookup model.
#class EmbeddingPredictor():
#    def __init__(self, embeddings, labels, n_neighbors=5):
#        self.embeddings = embeddings
#        self.labels = labels
#        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
#        self.model.fit(self.embeddings, self.labels) 

#    def __len__(self):
#        return len(self.embeddings)

#    def predict_closest_label(self, embedding):
#        return self.model.predict_proba([embedding])

#    def predict_closest_labels(self, embedding_list):
#        return self.model.predict_proba(embedding_list)

#    def get_distance(self, a, b):
#        return np.linalg.norm(a-b)**2
