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

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import cv2
import matplotlib as mp
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import face_embedding
import datautils
import align_dlib


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter, 
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p

def plot_maps(layer, maps):
    nrof_filters = maps.shape[0]
    fig = plt.figure(1, figsize=(10, 10))
    title = '{} Map'.format(layer)
    fig.canvas.set_window_title(title.replace('/', '_').replace(':', '_'))
    plt.title(layer)
    plt.axis('off')
    nrof_columns = 6
    nrof_rows = nrof_filters // nrof_columns + 1
    grid = ImageGrid(fig, 111, nrows_ncols=(nrof_rows, nrof_columns), axes_pad=0)
    for filter_idx in range(nrof_filters):
        f_activ = maps[filter_idx] 
        grid[filter_idx].imshow(f_activ, interpolation='nearest')
        grid[filter_idx].axis('off')
        grid[filter_idx].set_xticks([])
        grid[filter_idx].set_yticks([])
    plt.show()

def plot_activations(layer, activations):
    nrof_filters = activations.shape[2]
    fig = plt.figure(1, figsize=(10, 10))
    fig.canvas.set_window_title(layer.replace('/', '_').replace(':', '_'))
    plt.title(layer)
    plt.axis('off')
    nrof_columns = 6
    nrof_rows = nrof_filters // nrof_columns + 1
    grid = ImageGrid(fig, 111, nrows_ncols=(nrof_rows, nrof_columns), axes_pad=0)
    for filter_idx in range(nrof_filters):
        f_activ = activations[:, :, filter_idx] 
        f_max = np.max(f_activ)
        if f_max > 0:
            f_activ /= f_max
        grid[filter_idx].imshow(f_activ, interpolation='nearest', cmap='gray')
        grid[filter_idx].axis('off')
        grid[filter_idx].set_xticks([])
        grid[filter_idx].set_yticks([])
    plt.show()

def plot_activation(layer, activations, filter_idx):
    fig = plt.figure(1, figsize=(5, 5))
    title = '{} Filter {}'.format(layer, filter_idx)
    fig.canvas.set_window_title(title.replace('/', '_').replace(':', '_'))
    plt.title(title)
    plt.axis('off')
    activation = activations[:, :, filter_idx]
    f_max = np.max(activation)
    if f_max > 0:
        activation /= f_max
    plt.imshow(activation, interpolation='nearest', cmap='gray')
    plt.show()

def plot_map(layer, map, filter_idx):
    fig = plt.figure(1, figsize=(5, 5))
    title = '{} Filter Map {}'.format(layer, filter_idx)
    fig.canvas.set_window_title(title.replace('/', '_').replace(':', '_'))
    plt.title(title)
    plt.axis('off')
    plt.imshow(map, interpolation='nearest')
    plt.show()

def plot_image(image):
    plt.figure(1, figsize=(5, 5))  
    plt.subplot(1, 1, 1)
    plt.title('image')
    plt.imshow(image, interpolation='nearest')
    plt.axis('off')
    plt.show()

def rgb2gray(rgb):
    gray = np.copy(rgb).astype(float) / 255
    mean = np.mean(gray, axis=-1, keepdims=1)
    gray[:] = mean
    gray /= np.max(gray)
    return gray

#https://www.tensorflow.org/api_guides/python/nn#Convolution
def calculate_output_size(input_size, kernel_size, strides, padding):
    #https://www.tensorflow.org/api_guides/python/nn#Convolution
    input_i, input_j = input_size
    kernel_i, kernel_j = kernel_size
    stride_i, stride_j = strides
    if padding == 'SAME':
        output_i = int(math.ceil(float(input_i) / float(stride_i)))
        output_j = int(math.ceil(float(input_j) / float(stride_j)))
        return output_i, output_j
    elif padding == 'VALID':
        output_i = int(math.ceil(float(input_i - kernel_i + 1) / float(stride_i)))
        output_j  = int(math.ceil(float(input_j - kernel_j + 1) / float(stride_j)))
        return output_i, output_j
    else:
        raise Exception('Unknown padding type: {}'.format(padding))

#def calculate_input_size(output_size, kernel_size, strides, padding):
#    #https://www.tensorflow.org/api_guides/python/nn#Convolution
#    input_i, input_j = output_size
#    kernel_i, kernel_j = kernel_size
#    stride_i, stride_j = strides
#    if padding == 'SAME':
#        output_i = input_i * stride_i
#        output_j = input_j * stride_j
#        return output_i, output_j
#    elif padding == 'VALID':
#        output_i = (input_i + kernel_i) * stride_i
#        output_j = (input_j + kernel_j - 2) * stride_j
#        return output_i, output_j
#    else:
#        raise Exception('Unknown padding type: {}'.format(padding))


#https://www.tensorflow.org/api_guides/python/nn#Convolution
def calculate_kernel_pad(padding, input_size, kernel_size, strides):
    input_i, input_j = input_size
    kernel_i, kernel_j = kernel_size
    stride_i, stride_j = strides
    if padding == 'SAME':
        if (input_i % stride_i == 0):
            pad_i = int(max(kernel_i - stride_i, 0))
        else:
            pad_i = int(max(kernel_i - (input_i % stride_i), 0))
        if (input_j % stride_j == 0):
            pad_j = int(max(kernel_j - stride_j, 0))
        else:
            pad_j = int(max(kernel_j - (input_j % stride_j), 0))
        return pad_i, pad_j
    elif padding == 'VALID':
        return 0, 0
    else:
        raise Exception('Unknown padding type: {}'.format(padding))

def calculate_kernel_split(pad_size):
    pad_i, pad_j = pad_size
    pad_top = pad_i // 2
    pad_bottom = pad_i - pad_top
    pad_left = pad_j // 2
    pad_right = pad_j - pad_left
    return pad_top, pad_bottom, pad_left, pad_right

def kernel_unpad(input_array, padding, input_size, kernel_size, strides):
    pad_size = calculate_kernel_pad(padding, input_size, kernel_size, strides)
    pad_top, pad_bottom, pad_left, pad_right = calculate_kernel_split(pad_size)
    if padding == 'SAME':
        input_array = input_array[:, pad_top:input_array.shape[1]-pad_bottom, pad_left:input_array.shape[2]-pad_right]
    return input_array

def kernel_pad(input_array, padding, input_size, kernel_size, strides):
    pad_size = calculate_kernel_pad(padding, input_size, kernel_size, strides)
    pad_top, pad_bottom, pad_left, pad_right = calculate_kernel_split(pad_size)
    if padding == 'SAME':
        input_array = np.pad(input_array, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant')
    return input_array

def tf_kernel_pad(input_tensor, padding, input_size, kernel_size, strides):
    pad_size = calculate_kernel_pad(padding, input_size, kernel_size, strides)
    pad_top, pad_bottom, pad_left, pad_right = calculate_kernel_split(pad_size)
    if padding == 'SAME':
        input_tensor = tf.pad(input_tensor, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    return input_tensor

def tf_kernel_unpad(input_tensor, padding, input_size, kernel_size, strides):
    pad_size = calculate_kernel_pad(padding, input_size, kernel_size, strides)
    pad_top, pad_bottom, pad_left, pad_right = calculate_kernel_split(pad_size)
    if padding == 'SAME':
        input_tensor = input_tensor[:, pad_top:input_tensor.shape[1]-pad_bottom, pad_left:input_tensor.shape[2]-pad_right]
    return input_tensor

def get_layer_name(layer_out_name):
    return '/'.join(layer_out_name.split('/')[:-1])

def find_branch_input(branch_output_name, depth):
    branch_name = branch_output_name.split('/')[-3]
    # remove :0
    current_operation = emb_model.get_operation(branch_output_name[:-2])
    # work on quicker way to do this, lots of unnecessary traversal
    while branch_name in current_operation.name:
        current_tensor_name = current_operation.inputs[0].name[:-2]
        current_operation = emb_model.get_operation(current_tensor_name)
        depth += 1
    return current_operation.name, depth


def build_transpose_conv(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_transpose_tensor, stop_layer_out_name):
    layer_name = get_layer_name(layer_out_name)
    print('conv layer: {} at depth {}'.format(layer_name, depth))

    weights_tensor = emb_model.get_tensor('{}/{}'.format(layer_name, 'conv/kernel/read:0'))
    biases_tensor = emb_model.get_tensor('{}/{}'.format(layer_name, 'conv/bias/read:0'))

    #TODO check if there's batch normalization or not
    #TODO or do this in a seperate recursive build_transpose_bn
    try:
        beta_tensor = emb_model.get_tensor('{}/{}'.format(layer_name, 'bn/beta/read:0'))
        gamma_tensor = emb_model.get_tensor('{}/{}'.format(layer_name, 'bn/gamma/read:0'))
        population_mean_tensor = emb_model.get_tensor('{}/{}'.format(layer_name, 'bn/moving_mean/read:0'))
        population_variance_tensor = emb_model.get_tensor('{}/{}'.format(layer_name, 'bn/moving_variance/read:0'))
        is_batch_normalized = True
    except:
        is_batch_normalized = False

    if depth==0 or runnning_transpose_tensor is None:
        activ_tensor = tf.expand_dims(emb_model.get_tensor('{}/{}'.format(layer_name, 'activ:0'))[:, :, :, feature_idx], axis=-1)
        weights_tensor = tf.expand_dims(weights_tensor[:, :, :, feature_idx], axis=-1)
        biases_tensor = biases_tensor[feature_idx]
        if is_batch_normalized:
            beta_tensor = beta_tensor[feature_idx]
            gamma_tensor = gamma_tensor[feature_idx]
            population_mean_tensor = population_mean_tensor[feature_idx]
            population_variance_tensor = population_variance_tensor[feature_idx]
    else:
        activ_tensor = runnning_transpose_tensor

    in_operation = emb_model.get_operation('{}/{}'.format(layer_name, 'conv/Conv2D'))

    output_shape = tf.shape(in_operation.inputs[0])
    in_attr = in_operation.node_def.attr
    strides = (in_attr['strides'].list.i[0], in_attr['strides'].list.i[1], in_attr['strides'].list.i[2], in_attr['strides'].list.i[3])
    padding = in_attr['padding'].s.decode('ascii')
    
    activ_tensor = tf.nn.relu(activ_tensor)
    zero_epsilon = 0.0
    positive_activ_comparison = tf.greater(activ_tensor, zero_epsilon)
    
    if is_batch_normalized:
        bn_epsilon = 0.001
        activ_tensor = (((activ_tensor - beta_tensor) * (tf.sqrt(population_variance_tensor + bn_epsilon))) / (gamma_tensor)) + population_mean_tensor
        
    activ_tensor = tf.subtract(activ_tensor, biases_tensor)
    
    # denormalize correction for zero activations
    # set values to zero which had zero activations. 
    # This would not be necessary if there was not batch normalization, since
    # these values would already be zero from the activations, but 
    # because of batch norm I need to set the unnormalized values to zero where
    # the activations were zero, because otherwise the unnormalized values would be the zero
    # normalized value, and would not be zero during the deconvolution.
    activ_tensor = tf.where(positive_activ_comparison, activ_tensor, tf.zeros_like(activ_tensor))

    #transpose_tensor = tf.nn.conv2d_transpose(activ_tensor, weights_tensor, output_shape, strides, padding)
    #print(debug_run(tf.shape(transpose_tensor)))
    # use novel upscaling technique outlined here:
    # https://distill.pub/2016/deconv-checkerboard/
    # this avoids checkerboarding and looks much nicer, also more flexible for alternative upscaling options.
    # see 
    #TODO: "For best results, use tf.pad() before doing convolution with tf.nn.conv2d() to avoid boundary artifacts"
    #TODO try other methods
    transpose_tensor = tf.image.resize_images(activ_tensor, (output_shape[1], output_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    weights_transpose_tensor = tf.transpose(weights_tensor, [0, 1, 3, 2])
    transpose_tensor = tf.nn.conv2d(transpose_tensor, weights_transpose_tensor, strides, padding)
    #scale up to correct size
    transpose_tensor = tf.image.resize_images(transpose_tensor, (output_shape[1], output_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if layer_name == last_layer:
        prev_out_name = 'input'

        positive_img_comparison = tf.greater(transpose_tensor, zero_epsilon)

        # calculate mean and var over i, j, color
        input_mean, input_var = tf.nn.moments(emb_model.image_batch, axes=[1, 2, 3])
        input_std = tf.sqrt(input_var)
        input_shape = tf.shape(transpose_tensor)
        input_size = input_shape[1] * input_shape[2] * input_shape[3]
        input_std_adj_n = 1.0 / tf.sqrt(tf.cast(input_size, tf.float32))
        input_std_adj = tf.maximum(input_std, input_std_adj_n)
        transpose_tensor = (transpose_tensor * input_std_adj) + input_mean

        # denormalize correction for zero activations
        transpose_tensor = tf.where(positive_img_comparison, transpose_tensor, tf.zeros_like(transpose_tensor))

        transpose_tensor = tf.cast(tf.round(transpose_tensor), tf.uint8)
    else:
        prev_out_name = in_operation.inputs[0].name[:-2]
        
    return build_transpose_graph(emb_model, prev_out_name, last_layer, feature_idx, depth, transpose_tensor, stop_layer_out_name)

def build_transpose_maxpool(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_transpose_tensor, stop_layer_out_name):
    layer_name = get_layer_name(layer_out_name)
    print('maxpool layer: {} at depth {}'.format(layer_name, depth))

    if depth==0 or runnning_transpose_tensor is None:
        activ_tensor = tf.expand_dims(emb_model.get_tensor('{}/{}'.format(layer_name, 'MaxPool:0'))[:, :, :, feature_idx], axis=-1)
    else:
        activ_tensor = runnning_transpose_tensor

    in_operation = emb_model.get_operation('{}/{}'.format(layer_name, 'MaxPool'))

        
    output_shape = tf.shape(in_operation.inputs[0])
    in_attr = in_operation.node_def.attr
    strides = (in_attr['strides'].list.i[0], in_attr['strides'].list.i[1], in_attr['strides'].list.i[2], in_attr['strides'].list.i[3])
    padding = in_attr['padding'].s.decode('ascii')
    ksize = (in_attr['ksize'].list.i[1], in_attr['ksize'].list.i[2])
    assert ksize[0] == ksize[1]
    assert strides[1] == strides[2]

    #TODO try other methods
    transpose_tensor = tf.image.resize_images(activ_tensor, (output_shape[1], output_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if layer_name == last_layer:
        prev_out_name = 'input'
    else:
        prev_out_name = in_operation.inputs[0].name[:-2]

    return build_transpose_graph(emb_model, prev_out_name, last_layer, feature_idx, depth, transpose_tensor, stop_layer_out_name)

def build_transpose_concat(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_transpose_tensor, stop_layer_out_name):
    out_name = 'concat'
    print('{}: {} at depth {}'.format(out_name, layer_out_name, depth))
    layer_name = get_layer_name(layer_out_name)
    out_operation = emb_model.get_operation('{}/{}'.format(layer_name, out_name))
    
    # ignore setting this to some value if depth==0
    activ_tensor = runnning_transpose_tensor
    
    if activ_tensor is None:
        raise Exception('Cannot partition feature around concat.')

    branch_output_names = [str(x.name) for x in out_operation.inputs][:-1]

    branch_input_name = None
    branch_input_max_depth = depth
    for branch_output_name in branch_output_names:
        current_branch_input_name, current_branch_input_depth = find_branch_input(branch_output_name, depth)
        branch_input_max_depth = max(current_branch_input_depth, branch_input_max_depth)
        if branch_input_name is not None:
           assert branch_input_name == current_branch_input_name
        else:
            branch_input_name = current_branch_input_name
    
    print(' - Branch final layer: {}'.format(branch_input_name))

    branch_transpose = None
    branch_activ_idx = 0
    for branch_idx, branch_output_name in enumerate(branch_output_names):
        print('branch {}:'.format(branch_idx))
        # note: depths become useless at this point because each branch has 
        # different depths back to the same target node
        current_branch_feature_size = tf.shape(emb_model.get_tensor(branch_output_name))[3]
        # slice the concatenated tensor feature-wise for inputs
        current_branch_activ_tensor = activ_tensor[:, :, :, branch_activ_idx:branch_activ_idx+current_branch_feature_size]
        current_branch_transpose = build_transpose_graph(
            emb_model, branch_output_name, last_layer, feature_idx, 
            depth+1, current_branch_activ_tensor, branch_input_name)
        if branch_transpose is None:
            branch_transpose = current_branch_transpose
            branch_activ_idx = current_branch_feature_size
        else:
            branch_transpose = tf.add(branch_transpose, current_branch_transpose)
            branch_activ_idx = branch_activ_idx + current_branch_feature_size
        print('branch {} complete'.format(branch_idx))

    print('concat branch complete')
    return build_transpose_graph(emb_model, branch_input_name, last_layer, feature_idx, branch_input_max_depth, branch_transpose, stop_layer_out_name)

def build_transpose_mul(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_transpose_tensor, stop_layer_out_name):
    out_name = 'mul'
    print('{}: {} at depth {}'.format(out_name, layer_out_name, depth))
    layer_name = get_layer_name(layer_out_name)
    out_operation = emb_model.get_operation('{}/{}'.format(layer_name, out_name))

    if depth == 0 or runnning_transpose_tensor is None:
        activ_tensor = tf.expand_dims(emb_model.get_tensor('{}/{}'.format(layer_name, '{}:0'.format(out_name)))[:, :, :, feature_idx], axis=-1)
    else:
        activ_tensor = runnning_transpose_tensor

    input_name = out_operation.inputs[1].name[:-2]
    factor_tensor = out_operation.inputs[0]
    transpose_tensor = tf.divide(activ_tensor, factor_tensor)
    return build_transpose_graph(emb_model, input_name, last_layer, feature_idx, depth, transpose_tensor, stop_layer_out_name)

def build_transpose_add(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_transpose_tensor, stop_layer_out_name):
    out_name = 'add'
    print('{}: {} at depth {}'.format(out_name, layer_out_name, depth))
    layer_name = get_layer_name(layer_out_name)
    out_operation = emb_model.get_operation('{}/{}'.format(layer_name, out_name))

    if depth == 0 or runnning_transpose_tensor is None:
        activ_tensor = tf.expand_dims(emb_model.get_tensor('{}/{}'.format(layer_name, '{}:0'.format(out_name)))[:, :, :, feature_idx], axis=-1)
    else:
        activ_tensor = runnning_transpose_tensor

    # this is the skip connection
    skip_input = out_operation.inputs[0]
    skip_input_name = skip_input.name[:-2]

    # this is the convolutional section of the residual convolution
    layer_input_name = out_operation.inputs[1].name[:-2]
    # subtract off the skip connection activations
    layer_input = tf.subtract(activ_tensor, skip_input)
    # run through the graph up until the skip connection
    transpose_tensor = build_transpose_graph(emb_model, layer_input_name, last_layer, feature_idx, depth, layer_input, skip_input_name)
    # add the skip activations back to the projection
    transpose_tensor = tf.add(transpose_tensor, skip_input)
    
    #TODO depth value will be wildly inaccurate here
    return build_transpose_graph(emb_model, skip_input_name, last_layer, feature_idx, depth, transpose_tensor, stop_layer_out_name)

def build_transpose_relu(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_transpose_tensor, stop_layer_out_name):
    out_name = 'Relu'
    print('{}: {} at depth {}'.format(out_name, layer_out_name, depth))
    layer_name = get_layer_name(layer_out_name)
    out_operation = emb_model.get_operation('{}/{}'.format(layer_name, out_name))

    if depth == 0 or runnning_transpose_tensor is None:
        activ_tensor = tf.expand_dims(emb_model.get_tensor('{}/{}'.format(layer_name, '{}:0'.format(out_name)))[:, :, :, feature_idx], axis=-1)
    else:
        activ_tensor = runnning_transpose_tensor

    input_name = out_operation.inputs[0].name[:-2]
    transpose_tensor = tf.nn.relu(activ_tensor)
    return build_transpose_graph(emb_model, input_name, last_layer, feature_idx, depth, transpose_tensor, stop_layer_out_name)

def build_transpose_graph(emb_model, layer_out_name, last_layer, feature_idx, depth=-1, runnning_transpose_tensor=None, stop_layer_out_name='input'):
    depth += 1
    # at end of recursive depth
    if layer_out_name == stop_layer_out_name:
        return runnning_transpose_tensor

    if 'conv' in layer_out_name:
        return build_transpose_conv(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_transpose_tensor, stop_layer_out_name)
    elif 'maxpool' in layer_out_name:
        return build_transpose_maxpool(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_transpose_tensor, stop_layer_out_name)
    elif 'concat' in layer_out_name:
        return build_transpose_concat(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_transpose_tensor, stop_layer_out_name)
    elif 'mul' in layer_out_name:
        return build_transpose_mul(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_transpose_tensor, stop_layer_out_name)
    elif 'add' in layer_out_name:
        return build_transpose_add(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_transpose_tensor, stop_layer_out_name)
    elif 'Relu' in layer_out_name:
        return build_transpose_relu(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_transpose_tensor, stop_layer_out_name)
    else:
        raise Exception('Unknown graph element: '.format(layer_out_name))


def inverse_extract_image_patches(patches, strides, padding, output_shape):
    # TODO DELETE THIS if you can... or give up 
    patches_array, (batch_size, output_i, output_j, _)  = debug_run([patches, output_shape])
    _, input_i, input_j, kernel_i, kernel_j, output_c = patches_array.shape
    _, stride_i, stride_j, _ = strides
    # [1, input_i, input_j, input_c]
    output = np.zeros([batch_size, output_i, output_j, output_c], dtype=float)
    output = kernel_pad(output, padding, (input_i, input_j), (kernel_i, kernel_j), (stride_i, stride_j))
    # TODO: IF I can find a way to efficiently do this op in tensorflow I will be very happy
    # TODO: try one more time tomorrow...
    #patches_shape = tf.shape(patches)
    #patches_shape = patches.get_shape().as_list()
    #patches_shape[0] = tf.shape(patches)[0]
    #test_patches = tf.reshape(patches, tf.stack([patches_shape[0], patches_shape[1], patches_shape[2], 
    #                                    patches_shape[3] * patches_shape[4] * patches_shape[5]]))

    #test_patches = tf_kernel_pad(test_patches, padding, (input_i, input_j), (kernel_i, kernel_j), (stride_i, stride_j))
    #TODO an attempt was made, but still no good...
    #TODO might have better luck shuffling indexes around
    #test_kernel = tf.ones([kernel_i, kernel_j, output_c, kernel_i * kernel_j * output_c])
    #test_out = tf.nn.conv2d_transpose(test_patches, test_kernel, output_shape, strides, padding)
    #test_out = tf_kernel_unpad(test_out, padding, (input_i, input_j), (kernel_i, kernel_j), (stride_i, stride_j))
    #test = debug_run(test_out)
    #print(test.shape)
    #print(test[0, 0, 0, 0])
    
    print(patches.get_shape().as_list())
    for i in range(input_i):
        for j in range(input_j):
            out_i = i * stride_i
            out_j = j * stride_j
            output[:, out_i:out_i+kernel_i, out_j:out_j+kernel_j, :] += patches_array[:, i, j, :]

    output = kernel_unpad(output, padding, (input_i, input_j), (kernel_i, kernel_j), (stride_i, stride_j))
    #print(output.shape)
    #print(output[0, 0, 0, 0])
    output_score_map = tf.convert_to_tensor(output, dtype=tf.float32)
    print(output_score_map.get_shape().as_list())
    return output_score_map


def build_map_graph(emb_model, layer_out_name, last_layer, feature_idx, depth=-1, runnning_tensor=None, stop_layer_out_name='input'):
    depth += 1
    # at end of recursive depth
    if layer_out_name == stop_layer_out_name:
        return runnning_tensor

    if 'conv' in layer_out_name:
        return build_map_conv(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_tensor, stop_layer_out_name)
    elif 'maxpool' in layer_out_name:
        return build_map_maxpool(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_tensor, stop_layer_out_name)
    #elif 'concat' in layer_out_name:
    #    return build_map_concat(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_tensor, stop_layer_out_name)
    #elif 'mul' in layer_out_name:
    #    return build_map_mul(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_tensor, stop_layer_out_name)
    #elif 'add' in layer_out_name:
    #    return build_map_add(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_tensor, stop_layer_out_name)
    #elif 'Relu' in layer_out_name:
    #    return build_map_relu(emb_model, layer_out_name, last_layer, feature_idx, depth, runnning_tensor, stop_layer_out_name)
    else:
        raise Exception('Unknown graph element: '.format(layer_out_name))

    
def build_map_maxpool(emb_model, layer_out_name, last_layer, feature_idx, depth, prev_map_score, stop_layer_out_name):
    layer_name = get_layer_name(layer_out_name)
    print('maxpool layer: {} at depth {}'.format(layer_name, depth))

    if depth==0 or prev_map_score is None:
        input_score = tf.expand_dims(emb_model.get_tensor('{}/{}'.format(layer_name, 'MaxPool:0'))[:, :, :, feature_idx], axis=-1)
    else:
        input_score = prev_map_score

    in_operation = emb_model.get_operation('{}/{}'.format(layer_name, 'MaxPool'))

    output_shape = tf.shape(in_operation.inputs[0])
    in_attr = in_operation.node_def.attr
    strides = in_operation.get_attr('strides')
    padding = in_operation.get_attr('padding')
    ksize = in_operation.get_attr('ksize')
    padding = in_operation.get_attr('padding')
    #padding = in_attr['padding'].s.decode('ascii')

    #TODO can use actual max mapping here since I have values on both sides.
    output_score = tf.image.resize_images(input_score, (output_shape[1], output_shape[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if layer_name == last_layer:
        prev_out_name = 'input'
    else:
        prev_out_name = in_operation.inputs[0].name[:-2]

    return build_map_graph(emb_model, prev_out_name, last_layer, feature_idx, depth, output_score, stop_layer_out_name)




def build_map_conv(emb_model, layer_out_name, last_layer, feature_idx, depth, prev_map_score, stop_layer_out_name):
    layer_name = get_layer_name(layer_out_name)
    print('conv layer: {} at depth {}'.format(layer_name, depth))
    
    kernel_tensor = emb_model.get_tensor('{}/{}'.format(layer_name, 'conv/kernel/read:0'))
    biases_tensor = emb_model.get_tensor('{}/{}'.format(layer_name, 'conv/bias/read:0'))

    try:
        beta_tensor = emb_model.get_tensor('{}/{}'.format(layer_name, 'bn/beta/read:0'))
        gamma_tensor = emb_model.get_tensor('{}/{}'.format(layer_name, 'bn/gamma/read:0'))
        population_mean_tensor = emb_model.get_tensor('{}/{}'.format(layer_name, 'bn/moving_mean/read:0'))
        population_variance_tensor = emb_model.get_tensor('{}/{}'.format(layer_name, 'bn/moving_variance/read:0'))
        is_batch_normalized = True
    except:
        is_batch_normalized = False

    if depth==0 or prev_map_score is None:
        input_score = tf.expand_dims(emb_model.get_tensor('{}/{}'.format(layer_name, 'activ:0'))[:, :, :, feature_idx], axis=-1)
        kernel_tensor = tf.expand_dims(kernel_tensor[:, :, :, feature_idx], axis=-1)
        biases_tensor = biases_tensor[feature_idx]
        if is_batch_normalized:
            beta_tensor = beta_tensor[feature_idx]
            gamma_tensor = gamma_tensor[feature_idx]
            population_mean_tensor = population_mean_tensor[feature_idx]
            population_variance_tensor = population_variance_tensor[feature_idx]
    else:
        input_score = prev_map_score
        
    # input_score:
    # When we are at the final output layer, this is the same as the activations.
    # When we are at any other layer, input_score represents how much the inputs of the next layer influenced
    # the activations of the next layer.
    
    #TODO add epsilon
    input_score = input_score / tf.reduce_max(input_score)

    in_operation = emb_model.get_operation('{}/{}'.format(layer_name, 'conv/Conv2D'))
    #input_shape = in_operation.inputs[0].get_shape().as_list()
    in_attr = in_operation.node_def.attr
    #TODO clean this up...
    #TODO replace with op.get_attr('ksizes')
    strides = (in_attr['strides'].list.i[0], in_attr['strides'].list.i[1], in_attr['strides'].list.i[2], in_attr['strides'].list.i[3])
    padding = in_attr['padding'].s.decode('ascii')
    #TODO clean this up...
    dilations = (in_attr['dilations'].list.i[0], in_attr['dilations'].list.i[1], in_attr['dilations'].list.i[2], in_attr['dilations'].list.i[3])
    input_score = tf.nn.relu(input_score)
    zero_epsilon = 0.0
    #positive_activ_comparison = tf.greater(input_score, zero_epsilon)
    
    #if is_batch_normalized:
    #    bn_epsilon = 0.001
    #    input_score = (((input_score - beta_tensor) * (tf.sqrt(population_variance_tensor + bn_epsilon))) / (gamma_tensor)) + population_mean_tensor
        
    #input_score = tf.subtract(input_score, biases_tensor)
    
    # denormalize correction for zero activations
    # set values to zero which had zero activations. 
    # This would not be necessary if there was not batch normalization, since
    # these values would already be zero from the activations, but 
    # because of batch norm I need to set the unnormalized values to zero where
    # the activations were zero, because otherwise the unnormalized values would be the zero
    # normalized value, and would not be zero during the deconvolution.
    #input_score = tf.where(positive_activ_comparison, input_score, tf.zeros_like(input_score))

    #TODO 
    # 1. do sub convolution operation in a forward sense
    #https://www.tensorflow.org/api_guides/python/nn#Convolution
    # [1, input_i, input_j, input_c]
    input_tensor = in_operation.inputs[0]
    batch_size_tensor = tf.shape(input_tensor)[0]
    #TODO read this from graph and not from runtime
    kernel_size = kernel_tensor.get_shape().as_list()
    input_c = kernel_size[2]
    output_c = kernel_size[3]
    ksize = [1, kernel_size[0], kernel_size[1], 1]
    kernel_i, kernel_j = ksize[1], ksize[2]

    # [1, output_i, output_j, kernel_i * kernel_j * input_c]
    sub_conv_patches_op = tf.extract_image_patches(input_tensor, ksize, strides, dilations, padding)
    #sub_conv_patches_op_shape = tf.shape(sub_conv_patches_op)
    sub_conv_patches_op_shape = sub_conv_patches_op.get_shape().as_list()
    sub_conv_patches_op_shape[0] = batch_size_tensor
    sub_conv_patches = tf.reshape(
        sub_conv_patches_op, 
        shape=tf.stack([sub_conv_patches_op_shape[0], sub_conv_patches_op_shape[1], sub_conv_patches_op_shape[2], 
               ksize[1], ksize[2], input_c]))
    
    # [kernel_i, kernel_j, input_c, output_c]
    # kernel_tensor (defined above)

    # [1, output_i, output_j, kernel_i, kernel_j, input_c]
    # sub_conv_patches (defined above)

    # this represents the results we get from actually doing all the convolutional patch
    # multiplications, but not reducing them as a sum to a single value for each patch.
    # [1, output_i, output_j, output_c, kernel_i, kernel_j, input_c] -> output_score_list
    # overview: we will convolve over the input to this layer and calculate the partial convolutions
    # (same as normal convolution, but avoid reducing the [kernel_i, kernel_j, input_c] sum (dot product)
    # and first calculate a softmax ratio on these partial convolutions to produce a ratio of magnitudes.
    # we then multiply these ratios by the activations they produced. This effectively scores each 
    # [kernel_i, kernel_j, input_c] input value. This scoring produces the following shape:
    # [1, output_i, output_j, output_c, kernel_i, kernel_j, input_c]
    # We then take these scores and sum over output_c, producing the following shape:
    # [1, output_i, output_j, kernel_i, kernel_j, input_c]

    # [1, output_i, output_j, output_c, kernel_i, kernel_j, input_c]
    output_c_score_list = []
    for output_c_idx in range(output_c):
        output_c_sub_conv = tf.multiply(sub_conv_patches, kernel_tensor[:, :, :, output_c_idx])
        #output_c_shape = tf.shape(output_c_sub_conv)
        output_c_shape = output_c_sub_conv.get_shape().as_list()
        output_c_shape[0] = batch_size_tensor
        output_c_sub_conv_flat = tf.reshape(
            output_c_sub_conv, 
            shape=tf.stack([output_c_shape[0], output_c_shape[1], output_c_shape[2], 
                   output_c_shape[3] * output_c_shape[4] * output_c_shape[5]]))
        # filter out negative weight * input values, since we are looking for activation influencing input pixels
        # and negative multiplications should not contribute
        #positive_comparison = tf.greater(output_c_sub_conv_flat, 0.0)
        #output_c_sub_conv_flat = tf.where(positive_comparison, output_c_sub_conv_flat, tf.zeros_like(output_c_sub_conv_flat))
        output_c_sub_conv_flat = tf.nn.relu(output_c_sub_conv_flat)

        # 2. calculate softmax for each sub convolution over each kernel's kernel_i, kernel_j, input_c
        output_c_softmax_flat = tf.nn.softmax(output_c_sub_conv_flat, axis=-1)
        output_c_softmax_flat_t = tf.transpose(output_c_softmax_flat, [3, 0, 1, 2])
        # 3. multiply these ratios for each sub convolution by the corresponding input score
        output_c_score_flat_t = tf.multiply(output_c_softmax_flat_t, input_score[:, :, :, output_c_idx])
        output_c_score_flat = tf.transpose(output_c_score_flat_t, [1, 2, 3, 0])
        output_c_score = tf.reshape(output_c_score_flat, output_c_shape)
        output_c_score_list.append(output_c_score)
        
    # [1, output_i, output_j, output_c, kernel_i, kernel_j, input_c]
    output_c_scores = tf.stack(output_c_score_list, axis=3)
    
    # [1, output_i, output_j, kernel_i, kernel_j, input_c]
    output_scores = tf.reduce_sum(output_c_scores, axis=[3])
    
    #output_scores_shape = [x if x is not None else tf.shape(output_scores)[idx] for idx, x in enumerate(output_scores.get_shape().as_list())]

    # [1, output_i, output_j, kernel_i * kernel_j * input_c]
    #output_scores_t = tf.reshape(
    #    output_scores, 
    #    shape=[output_scores_shape[0], output_scores_shape[1], output_scores_shape[2], 
    #           output_scores_shape[3] * output_scores_shape[4] * output_scores_shape[5]])
    

    # TODO: need to create this shape
    # essentially I need to create an inverse for extract_image_patches
    # there is an implementation of an inverse for gradients under tf.python.ops.array_grad._ExtractImagePatchesGrad
    # https://github.com/tensorflow/tensorflow/issues/6847
    # [1, input_i, input_j, input_c]
    output_score_map = inverse_extract_image_patches(output_scores, strides, padding, tf.shape(in_operation.inputs[0]))
    
    if layer_name == last_layer:
        prev_out_name = 'input'

        #positive_img_comparison = tf.greater(output_score_map, 0.0)

        # calculate mean and var over i, j, color
        #input_mean, input_var = tf.nn.moments(emb_model.image_batch, axes=[1, 2, 3])
        #input_std = tf.sqrt(input_var)
        #input_shape = tf.shape(output_score_map)
        #input_size = input_shape[1] * input_shape[2] * input_shape[3]
        #input_std_adj_n = 1.0 / tf.sqrt(tf.cast(input_size, tf.float32))
        #input_std_adj = tf.maximum(input_std, input_std_adj_n)
        #output_score_map = (output_score_map * input_std_adj) + input_mean
        #TODO add epsilon
        output_score_map = output_score_map / tf.reduce_max(output_score_map)
        # denormalize correction for zero activations
        #output_score_map = tf.where(positive_img_comparison, output_score_map, tf.zeros_like(output_score_map))
        output_score_map = tf.nn.relu(output_score_map)
        #output_score_map = tf.cast(tf.round(output_score_map), tf.uint8)

    else:
        prev_out_name = in_operation.inputs[0].name[:-2]
        
    return build_map_graph(emb_model, prev_out_name, last_layer, feature_idx, depth, output_score_map, stop_layer_out_name)


def unstandardize_img_tensor(input_img, standard_img):
        positive_img_comparison = tf.greater(standard_img, 0.0)

        # calculate mean and var over i, j, color
        input_mean, input_var = tf.nn.moments(input_img, axes=[1, 2, 3])
        input_std = tf.sqrt(input_var)
        input_shape = tf.shape(standard_img)
        input_size = input_shape[1] * input_shape[2] * input_shape[3]
        input_std_adj_n = 1.0 / tf.sqrt(tf.cast(input_size, tf.float32))
        input_std_adj = tf.maximum(input_std, input_std_adj_n)
        unstandardized_img = (standard_img * input_std_adj) + input_mean

        # denormalize correction for zero activations
        unstandardized_img = tf.where(positive_img_comparison, unstandardized_img, tf.zeros_like(unstandardized_img))

        unstandardized_img = tf.cast(tf.round(unstandardized_img), tf.uint8)
        return unstandardized_img


def unstandardize_img(input_img, output_standardized_img):
    img_mean = np.mean(input_img)
    img_std = np.std(input_img)
    img_std_adj_n = 1.0/np.sqrt(input_img.size)
    img_std_adj = max(img_std, img_std_adj_n)
    output_unstandardized_img = ((output_standardized_img * img_std_adj) + img_mean)
    return output_unstandardized_img


def debug_run(tensor):
    return emb_model.sess.run(tensor, feed_dict={emb_model.image_batch: test_img_batch})


if __name__ == '__main__':
    face_img_size = 96
    embedding_size = 128
    dlib_model = align_dlib.AlignDlib('shape_predictor_68_face_landmarks.dat')
    emb_model = face_embedding.EmbeddingModel(face_img_size, 'InceptionResNetV1Small-VGGFace2-v1-1000.pb')
    emb_model.load()

    test_img_path = 'D:/Google Drive/Code/Python/ResearchFaceNet/ResearchFaceNet/Inspect/Maxwell Weinzierl/00000004/00000004.jpg'
    test_img_array = datautils.process_image(test_img_path, face_img_size, dlib_model)
    test_img_batch = np.expand_dims(test_img_array, axis=0)
    #test_img_r = np.copy(test_img_batch)
    #test_img_r[:, :, :, 1] = 0
    #test_img_r[:, :, :, 2] = 0
    #test_img_g = np.copy(test_img_batch)
    #test_img_g[:, :, :, 0] = 0
    #test_img_g[:, :, :, 2] = 0
    #test_img_b = np.copy(test_img_batch)
    #test_img_b[:, :, :, 0] = 0
    #test_img_b[:, :, :, 1] = 0
    #plot_image(test_img_batch[0])
    #plot_image(test_img_r[0])
    #plot_image(test_img_g[0])
    #plot_image(test_img_b[0])

    #layers = ['cnn/stem/conv_1a_3x3/activ:0', 
    #          'cnn/stem/conv_2a_3x3/activ:0',
    #          'cnn/stem/conv_2b_3x3/activ:0',
    #          'cnn/stem/conv_3b_1x1/activ:0',
    #          'cnn/stem/conv_4a_3x3/activ:0',
    #          'cnn/stem/conv_4b_3x3/activ:0']

    #layer_name = 'cnn/stem/conv_1a_3x3/activ'
    #layer_name = 'cnn/inception_resnet_a/repeat_0/branch_2/conv_0c_3x3/activ'
    #layer_out_name = 'cnn/inception_resnet_a/repeat_0/Relu'
    #TODO there is an issue with either the add, mul, relu, or concat transpose graph visits.
    #TODO everything works well up until those. Further inspection is necessary
    layer_out_name = 'cnn/stem/conv_2a_3x3/activ'
    feature_idx = 1

    last_layer = 'cnn/stem/conv_1a_3x3'

    #layer_activ = emb_model.sess.run('{}:0'.format(layer_out_name), feed_dict={emb_model.image_batch: test_img_batch})[0]
    #plot_activations(layer_out_name, layer_activ)
    #plot_activation(layer_out_name, layer_activ, feature_idx)
    #plot_image(test_img_array)

    layer_filter_size = emb_model.sess.run(
        tf.shape(emb_model.get_tensor('{}:0'.format(layer_out_name)))[3], 
        feed_dict={emb_model.image_batch: test_img_batch})
    
    filter_range = range(layer_filter_size)
    #layer_activ = emb_model.sess.run('{}:0'.format(layer_out_name), feed_dict={emb_model.image_batch: test_img_batch})[0]
    #plot_activations(layer_out_name, layer_activ[:, :, :12])
    
    with emb_model.graph.as_default():
        final_map_tensor = build_map_graph(emb_model, layer_out_name, last_layer, feature_idx)[0]
        final_map = emb_model.sess.run(final_map_tensor, feed_dict={emb_model.image_batch: test_img_batch})
        plot_map(layer_out_name, final_map, feature_idx)

        #map_tensors = []
        #for filter_idx in filter_range:
        #    map_tensor = build_map_graph(emb_model, layer_out_name, last_layer, filter_idx)[0]
        #    map_tensors.append(map_tensor)
        #filter_map_tensor = tf.stack(map_tensors)
        #filter_map = emb_model.sess.run(filter_map_tensor, feed_dict={emb_model.image_batch: test_img_batch})
        #plot_maps(layer_out_name, filter_map)
        

        #transpose_tensor = build_transpose_graph(emb_model, layer_out_name, last_layer, feature_idx)[0]
        #transpose = emb_model.sess.run(transpose_tensor, feed_dict={emb_model.image_batch: test_img_batch})
        #plot_map(layer_out_name, transpose, feature_idx)

        #transpose_tensors = []
        #for filter_idx in filter_range:
        #    transpose_tensor = build_transpose_graph(emb_model, layer_out_name, last_layer, filter_idx)[0]
        #    transpose_tensors.append(transpose_tensor)
        #filter_transpose_tensor = tf.stack(transpose_tensors)
        #filter_transpose = emb_model.sess.run(filter_transpose_tensor, feed_dict={emb_model.image_batch: test_img_batch})
        #plot_maps(layer_out_name, filter_transpose)

