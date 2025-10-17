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
import os
import time
from abc import ABC, abstractmethod
from tensorflow.python.platform import gfile
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import graph_util
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers

import itertools
from functools import partial

import datautils
import model_layers


class FaceNetModel(ABC):
    def __init__(self, model_args):
        self.model_args = model_args
        self.name = model_args.model_name
        self.model_directory = model_args.model_directory
        self.image_size = model_args.image_size
        self.embedding_size = model_args.embedding_size
        self.logger = logging.getLogger(self.name)
        
        self.checkpoint_dir = os.path.join(self.model_directory, 'checkpoints')
        self.vali_checkpoint_dir = os.path.join(self.model_directory, 'validation_checkpoints')
        self.graph_filename = os.path.join(self.model_directory, self.name + '.pb')
        self.save_location = os.path.join(self.checkpoint_dir, self.name)
        self.vali_save_location = os.path.join(self.vali_checkpoint_dir, self.name)
    
    def get_nrof_trainable_parameters(self, log_info=False):
        with self.graph.as_default():
            #TODO ignore batch norm & optimizer parameters
            total_parameters = 0
            for variable in [x for x in tf.trainable_variables() if ('/bn/' not in x.name) and ('fcn_bn' not in x.name)]:
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
                if log_info:
                    self.logger.info('{} ({} *= {}) (total={})'.format(variable.name, shape, variable_parameters, total_parameters))
            return total_parameters

    def load(self, trainable=True):
        if not os.path.exists(self.model_directory):
           raise Exception('Unable to load model folder!')
        if trainable:
            self.graph = tf.Graph()
            with self.graph.as_default():
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(graph=self.graph, config=config)
                if os.path.exists(self.save_location + '.meta'):
                    self.logger.info('Loading graph meta file...')
                    self.saver = tf.train.import_meta_graph(self.save_location + '.meta')
                    self.logger.info('Restoring last checkpoint...')
                    self.saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_dir))
                    total_parameters = self.get_nrof_trainable_parameters()
                    self.logger.info('Total number of trainable parameters: {}'.format(total_parameters))
                else:
                    raise Exception('Unable to load graph!')
        else:
            if os.path.exists(self.graph_filename):
                self.graph = datautils.load_graph(self.graph_filename)
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(graph=self.graph, config=config)
            else:
                 raise Exception('Unable to load frozen graph!') 
        self.setup()

    def create(self):
        if not os.path.exists(self.model_directory):
            os.mkdir(self.model_directory)
        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.graph, config=config)
            self.logger.info('Building graph...')
            self.build()
            self.saver = tf.train.Saver(max_to_keep=3, save_relative_paths=True)
            self.vali_saver = tf.train.Saver(max_to_keep=3, save_relative_paths=True)
            if not os.path.exists(self.checkpoint_dir):
                self.logger.info('Building checkpoint folder...')
                os.mkdir(self.checkpoint_dir)
            if not os.path.exists(self.vali_checkpoint_dir):
                self.logger.info('Building validation checkpoint folder...')
                os.mkdir(self.vali_checkpoint_dir)
            self.logger.info('Initializing variables...')
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            total_parameters = self.get_nrof_trainable_parameters(log_info=True)
            self.logger.info('Total number of trainable parameters: {}'.format(total_parameters))
            self.logger.info('Saving meta graph...')
            self.saver.export_meta_graph(self.save_location + '.meta')

        self.setup()

    def unload(self):
        try:
            self.sess.close()
        except NameError:
            pass
        #TODO this may be unnecessary
        tf.reset_default_graph()
            
        


    def freeze(self):
        self.logger.info('Freezing graph...')
        checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir)
        input_checkpoint = checkpoint.model_checkpoint_path

        # We precise the file fullname of our freezed graph
        output_node_names=[]
        reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
        var_to_shape_map = reader.get_variable_to_shape_map()
        #TODO remove ADAGRAD/train prefixed vars
        for key in var_to_shape_map:
            if 'train/' in key:
                continue
            output_node_names.append(key)
            self.logger.info('tensor_name: {}'.format(key))

        save_ops = ['inputs/image_batch', 
                    'inputs/global_step', 
                    'inputs/prob_keep', 
                    'inputs/is_training', 
                    'outputs/embeddings']
        for key in save_ops:
            output_node_names.append(key)

        output_node_names=str(output_node_names)       
        output_node_names=output_node_names.replace('u\'','\'')
        output_node_names=output_node_names.replace('[','')
        output_node_names=output_node_names.replace(']','')
        output_node_names=output_node_names.replace('\'','')
        output_node_names=output_node_names.replace('\'','')
        output_node_names=output_node_names.replace(' ','')
        # issue with this:

        clear_devices = True

        saver = tf.train.import_meta_graph(self.save_location + '.meta', clear_devices=clear_devices)
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        with tf.Session() as sess:
            saver.restore(sess, input_checkpoint)
        
            output_graph_def = graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                input_graph_def, # The graph_def is used to retrieve the nodes 
                output_node_names.split(',')# The output node names are used to select the useful nodes
            ) 
            with tf.gfile.GFile(self.graph_filename, 'wb') as f:
                f.write(output_graph_def.SerializeToString())

            self.logger.info('{} ops in the final graph.'.format(len(output_graph_def.node)))
            self.logger.info('Graph frozen.')

    
    def build(self):
        # soft place as much on gpu as possible explicitly
        with tf.device('/device:GPU:0'):
            with tf.variable_scope('inputs'):
                
                # margin to enforce. only used for accuracy checks, as it isnt needed in the minimization step.
                margin = tf.placeholder(tf.float32, name='margin')

                learning_rate = tf.placeholder(tf.float32, name='learning_rate')

                prob_keep = tf.placeholder(tf.float32, name='prob_keep')
                is_training = tf.placeholder(tf.bool, name='is_training')

                batch_size = tf.placeholder(tf.int32, name='batch_size')
                global_step = tf.Variable(0, trainable=False, name='global_step')
                
                image_paths = tf.placeholder(tf.string, shape=[None, 3], name='image_paths')
                labels = tf.placeholder(tf.int64, shape=[None, 3], name='labels')

                batch_images, batch_labels, enqueue_op = datautils.read_data(image_paths, labels, self.image_size, batch_size, 
                                                                 shuffle=False, 
                                                                 num_threads=self.model_args.num_queue_threads, 
                                                                 queue_capacity=self.model_args.queue_capacity,
                                                                 random_flip=self.model_args.random_flip, 
                                                                 random_brightness=self.model_args.random_brightness, 
                                                                 random_contrast=self.model_args.random_contrast, 
                                                                 random_crop=self.model_args.random_crop)
                
                regularizer = tf.contrib.layers.l2_regularizer(scale=self.model_args.weight_decay)
                # Xavier initializer proved to work very poorly. Look into why it was so bad!
                #initializer = layers.xavier_initializer_conv2d()
                initializer = tf.truncated_normal_initializer(stddev=0.1)

            with tf.variable_scope('cnn'):
                # consider alternatives to relu
                activation_fn = tf.nn.relu
                cnn_final_layer = self.build_cnn(self.model_args, batch_images, is_training, prob_keep, regularizer, initializer, activation_fn)

                cnn_out = tf.layers.flatten(cnn_final_layer, name='cnn_out')

            with tf.variable_scope('fcn'):
                fcn_out = tf.layers.dense(inputs=cnn_out, 
                                          units=self.embedding_size, 
                                          activation=None, 
                                          kernel_initializer=initializer, 
                                          kernel_regularizer=regularizer,
                                          name='fcn_out')

                fcn_out = tf.layers.batch_normalization(fcn_out, training=is_training, momentum=0.995, epsilon=0.001, name='fcn_bn')

            with tf.variable_scope('outputs'):

                # [ BATCHSIZE x 1, EMBEDDINGSSIZE ] or 
                # [ BATCHSIZE x 3, EMBEDDINGSSIZE ] 
                embeddings = tf.nn.l2_normalize(fcn_out, axis=1, name='embeddings')
                
                
                # [ BATCHSIZE, 3, EMBEDDINGSSIZE ] 
                training_embeddings = tf.reshape(embeddings, [-1, 3, self.embedding_size], name='training_embeddings')
                
                anchor_embeddings, positive_embeddings, negative_embeddings = tf.unstack(training_embeddings, 3, 1)
                anchor_embeddings = tf.identity(anchor_embeddings, name='anchor_embeddings')
                positive_embeddings = tf.identity(positive_embeddings, name='positive_embeddings')
                negative_embeddings = tf.identity(negative_embeddings, name='negative_embeddings')

            with tf.variable_scope('loss'):
                reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')

                # consider using cosine distance ?
                #  [ BATCHSIZE ] 
                #
                pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor_embeddings, positive_embeddings)), axis=1, name='positive_distance')
                neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor_embeddings, negative_embeddings)), axis=1, name='negative_distance')
                       
                #Pairwise Triplet Loss (my idea): minimize [0.5 * (2 * (||a - p||^2 + margin) - ||a - n||^2 - ||p - n||^2)]
                if self.model_args.pairwise_triplet_loss:
                    pos_neg_dist = tf.reduce_sum(tf.square(tf.subtract(positive_embeddings, negative_embeddings)), axis=1, 
                                                           name='positive_negative_distance')
                    margin_loss = tf.identity(0.5 * ((2 * (pos_dist + margin)) - neg_dist - pos_neg_dist), name='margin_loss')
                #Triplet loss: minimize [||a - p||^2 + margin - ||a - n||^2]
                else:
                    margin_loss = tf.identity(pos_dist + margin - neg_dist, name='margin_loss')

                triplet_loss = tf.maximum(margin_loss, 0.0, name='triplet_loss')
                
                loss = tf.identity(tf.reduce_mean(triplet_loss) + reg_loss, name='loss')
                
                avg_pos_dist = tf.reduce_mean(pos_dist, name='avg_pos_dist')
                avg_neg_dist = tf.reduce_mean(neg_dist, name='avg_neg_dist')
                
                active_triplet_percent = tf.reduce_mean(tf.cast(tf.greater(triplet_loss, 0.0), tf.float32), name='active_triplet_percent')

            with tf.variable_scope('train'):
                learning_rate_decay = tf.train.exponential_decay(learning_rate, global_step,
                                                                self.model_args.learning_rate_decay_epochs * self.model_args.epoch_size, 
                                                                self.model_args.learning_rate_decay_factor, 
                                                                staircase=True,
                                                                name='learning_rate_decay')
                # consider alternatives to adagrad. so far adagrad has performed best for 
                # many different users online in similar projects, but RMSProp may also serve well.
                # definitely test RMSProp and ADAM at some point.
                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate_decay)
                #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_decay)

                gradients, variables = zip(*optimizer.compute_gradients(loss))

                # clip individually - faster but does not maintain relative grad ratio
                clipped_gradients = [tf.clip_by_norm(gradient, self.model_args.clip_norm) for gradient in gradients]
                # clip using global norm - slower but maintains ratio, also gives us global norm value
                #clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.model_args.clip_norm)
                # calculate infinite norm of gradients (max abs value)
                infinite_grad_norm = tf.reduce_max([tf.reduce_max(tf.abs(gradient)) for gradient in gradients], name='infinite_grad_norm')
                # make sure to perform updates to batch normalization during training steps.
                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step, name='train_op')


    def setup(self):
        # this is here in order to be able to recover tensors and ops when the model is loaded from file.
        self.enqueue_op = self.graph.get_operation_by_name('inputs/enqueue_op')

        self.image_paths = self.graph.get_tensor_by_name('inputs/image_paths:0')
        self.labels = self.graph.get_tensor_by_name('inputs/labels:0')
        self.image_batch = self.graph.get_tensor_by_name('inputs/image_batch:0')
        self.label_batch = self.graph.get_tensor_by_name('inputs/label_batch:0')
        self.batch_size = self.graph.get_tensor_by_name('inputs/batch_size:0')
        self.global_step = self.graph.get_tensor_by_name('inputs/global_step:0')

        self.margin = self.graph.get_tensor_by_name('inputs/margin:0')
        self.learning_rate = self.graph.get_tensor_by_name('inputs/learning_rate:0')
        self.prob_keep = self.graph.get_tensor_by_name('inputs/prob_keep:0')
        self.is_training = self.graph.get_tensor_by_name('inputs/is_training:0')

        self.embeddings = self.graph.get_tensor_by_name('outputs/embeddings:0')
        self.loss = self.graph.get_tensor_by_name('loss/loss:0')
        self.avg_pos_dist = self.graph.get_tensor_by_name('loss/avg_pos_dist:0')
        self.avg_neg_dist = self.graph.get_tensor_by_name('loss/avg_neg_dist:0')
        self.reg_loss = self.graph.get_tensor_by_name('loss/reg_loss:0')
        self.active_triplet_percent = self.graph.get_tensor_by_name('loss/active_triplet_percent:0')

        self.learning_rate_decay = self.graph.get_tensor_by_name('train/learning_rate_decay:0')
        self.infinite_grad_norm = self.graph.get_tensor_by_name('train/infinite_grad_norm:0')
        self.train_op = self.graph.get_operation_by_name('train/train_op')

        self.queue_size = self.graph.get_tensor_by_name('inputs/batch_join/fifo_queue_Size:0')

    def train(self, args):

        self.logger.info('Loading training image paths...')
        train_set, train_image_count = datautils.get_dataset(args.input_directory)

        self.logger.info('{} total training classes'.format(len(train_set)))
        self.logger.info('{} total training images'.format(train_image_count))
        

        self.logger.info('Loading training pairs...')
        train_pairs = datautils.generate_pairs(train_set, 
                                               args.nrof_train_pair_sample_classes, 
                                               args.nrof_train_pair_anchors, 
                                               args.nrof_train_pair_positive, 
                                               args.nrof_train_pair_negative)
        
        train_pairs_flat, train_actual_issame = datautils.flatten_pairs(train_pairs)

        self.logger.info('Loading lfw test pairs...')
        test_pairs = datautils.read_lfw_pairs(args.test_pairs_path)
        test_pairs_flat, test_actual_issame = datautils.get_lfw_paths(args.lfw_directory, test_pairs)
                
                
        self.logger.info('{} total train pairs'.format(len(train_pairs)))
        self.logger.info('{} total test pairs'.format(len(test_actual_issame)))
                
        self.logger.info('Initializing image queue runners...')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        
        step = self.sess.run(self.global_step)
        epoch = step // args.epoch_size
        
        self.logger.info('Creating log file...')
        # only need to write graph for visualization once
        file_graph = None if epoch > 0 else self.graph
        summary_writer = tf.summary.FileWriter(args.log_path, file_graph)
        if epoch == 0:
            # get initial accuracy numbers just for sanity check: ~50%
            self.evaluate('train', args, train_pairs_flat, train_actual_issame, step, summary_writer)
            self.evaluate('lfw', args, test_pairs_flat, test_actual_issame, step, summary_writer)
        while epoch < args.max_nrof_epochs:
            epoch = step // args.epoch_size
            epoch_start = time.time()
            self.train_epoch(args, train_set, epoch, summary_writer)
            step = self.sess.run(self.global_step)
            epoch_time = time.time() - epoch_start

            summary = tf.Summary()
            summary.value.add(tag='time/epoch', simple_value=epoch_time)
            summary_writer.add_summary(summary, step)
            
            self.logger.info('Saving graph variables...')
            self.saver.save(self.sess, self.save_location, global_step=step, write_meta_graph=False)
            
            self.evaluate('train', args, train_pairs_flat, train_actual_issame, step, summary_writer)
            self.evaluate('lfw', args, test_pairs_flat, test_actual_issame, step, summary_writer)

        try:
            coord.request_stop()
            coord.join(threads=threads)
        except RuntimeError as e:
            self.logger.error(e)


    def train_epoch(self, args, train_set, epoch, summary_writer):
        batch_number = 0
        while batch_number < args.epoch_size:
            batch_start_time = time.time()
            # Sample people randomly from the dataset
            self.logger.info('Sampling images and people...')
            sample_start_time = time.time()
            image_paths, num_per_class = datautils.sample_people(train_set, args.people_per_batch, args.images_per_person)
            sample_time = time.time() - sample_start_time
            self.logger.info('Running forward pass on sampled images...')
            fwd_start_time = time.time()

            nrof_examples = args.people_per_batch * args.images_per_person
            labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
            image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
            self.sess.run(self.enqueue_op, 
                          feed_dict={self.image_paths: image_paths_array, 
                                     self.labels: labels_array})

            emb_array = np.zeros((nrof_examples, self.embedding_size))
            nrof_mini_batches = int(np.ceil(nrof_examples / args.batch_size))
            for minibatch_idx in range(nrof_mini_batches):
                m_batch_size = min(nrof_examples - (minibatch_idx * args.batch_size), args.batch_size)
                emb, lab = self.sess.run([self.embeddings, self.label_batch], 
                                    feed_dict={self.batch_size: m_batch_size, 
                                               # use population statistics not batch statistics, intuition being why rely on batch
                                               # statistics for triplet selection when we can use pop stats? Problem is, if we use
                                               # batch statistics, then a larger percent of the triplets we pick may not violate the 
                                               # inequality after running through the model with the training batch statistics. Testing
                                               # seems to support this theory, where on average it seems like there's a good 10%-20% increase
                                               # in active triplets if we use population statistics for triplet selection.
                                               self.is_training: False, 
                                               self.prob_keep: 1.0})
                emb_array[lab, :] = emb
            forward_time = time.time() - fwd_start_time
            forward_queue_size = self.sess.run(self.queue_size)
            self.logger.info('Forward final queue size: {}'.format(forward_queue_size))

            # Select triplets based on the embeddings
            self.logger.info('Selecting suitable triplets for training...')
            trp_start_time = time.time()
            triplets, nrof_possible_triplets, nrof_triplets = \
                datautils.select_triplets(emb_array, num_per_class, 
                                          image_paths, args.people_per_batch, 
                                          args.margin, args.semi_hard_negatives, 
                                          args.pairwise_triplet_negatives)

            selection_time = time.time() - trp_start_time
            self.logger.info('(nrof_possible_triplets, nrof_triplets) = ({}, {})'.format(nrof_possible_triplets, nrof_triplets))

            # Perform training on the selected triplets
            nrof_mini_batches = int(np.ceil((nrof_triplets * 3)/args.batch_size))
            self.logger.info('Training next {} minibatches...'.format(nrof_mini_batches))

            triplet_paths = list(itertools.chain(*triplets))
            labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
            triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))
            self.sess.run(self.enqueue_op, 
                          feed_dict={self.image_paths: triplet_paths_array, 
                                     self.labels: labels_array})

            nrof_examples = len(triplet_paths)
            minibatch_idx = 0
            train_time = 0
            while minibatch_idx < nrof_mini_batches:
                minibatch_train_time = time.time()
                batch_size_m = min(nrof_examples - (minibatch_idx * args.batch_size), args.batch_size)
                loss, avg_p_d, avg_n_d, r_loss, actv_t_p, lr_d, i_g_n, step, _ = \
                    self.sess.run([self.loss, 
                                   self.avg_pos_dist,
                                   self.avg_neg_dist,
                                   self.reg_loss,
                                   self.active_triplet_percent,
                                   self.learning_rate_decay, 
                                   self.infinite_grad_norm, 
                                   self.global_step,
                                   self.train_op], 
                                  feed_dict={self.batch_size: batch_size_m, 
                                            self.learning_rate: args.learning_rate, 
                                            self.is_training: True,
                                            self.prob_keep: args.prob_keep,
                                            self.margin: args.margin})

                minibatch_train_time = time.time() - minibatch_train_time
                self.logger.info('Epoch: [{}][{}/{}] Time {:.3f} Loss {:.3f} [{:.3f} + {:.3f} - {:.3f}]'
                                 .format(epoch, batch_number + 1, args.epoch_size, minibatch_train_time, loss, avg_p_d, args.margin, avg_n_d))
                batch_number += 1
                minibatch_idx += 1
                train_time += minibatch_train_time
                b_summary = tf.Summary()
                b_summary.value.add(tag='loss/loss', simple_value=loss)
                b_summary.value.add(tag='loss/avg_positive_dist', simple_value=avg_p_d)
                b_summary.value.add(tag='loss/avg_negative_dist', simple_value=avg_n_d)
                b_summary.value.add(tag='loss/regularization_loss', simple_value=r_loss)
                b_summary.value.add(tag='loss/active_triplet_percent', simple_value=actv_t_p)
                b_summary.value.add(tag='loss/learning_rate_decay', simple_value=lr_d)
                b_summary.value.add(tag='loss/infinite_grad_norm', simple_value=i_g_n)
                b_summary.value.add(tag='time/train_minibatch', simple_value=minibatch_train_time)
                summary_writer.add_summary(b_summary, step)

            batch_time = time.time() - batch_start_time
            
            train_queue_size = self.sess.run(self.queue_size)
            self.logger.info('Train final queue size: {}'.format(train_queue_size))

            summary = tf.Summary()
            summary.value.add(tag='time/batch', simple_value=batch_time)
            summary.value.add(tag='time/sample', simple_value=sample_time)
            summary.value.add(tag='time/forward', simple_value=forward_time)
            summary.value.add(tag='time/selection', simple_value=selection_time)
            summary.value.add(tag='time/train_batch', simple_value=train_time)
            summary.value.add(tag='triplets/nrof_possible_triplets', simple_value=nrof_possible_triplets)
            summary.value.add(tag='triplets/nrof_triplets', simple_value=nrof_triplets)
            summary_writer.add_summary(summary, step)
        return step


    def evaluate(self, prefix, args, image_paths, actual_issame, step, summary_writer):
        start_time = time.time()
        # Run forward pass to calculate embeddings
        self.logger.info('[{}]: Running forward pass on images...'.format(prefix))
    
        nrof_images = len(actual_issame) * 2
        assert(len(image_paths)==nrof_images)
        labels_array = np.reshape(np.arange(nrof_images), (-1, 3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))

        self.sess.run(self.enqueue_op, {self.image_paths: image_paths_array, 
                                        self.labels: labels_array})

        emb_array = np.zeros((nrof_images, self.embedding_size))
        nrof_batches = int(np.ceil(nrof_images / args.batch_size))
        label_check_array = np.zeros((nrof_images,))
        for i in range(nrof_batches):
            m_batch_size = min(nrof_images - (i * args.batch_size), args.batch_size)
            emb, lab = self.sess.run([self.embeddings, self.label_batch], 
                                     feed_dict={self.batch_size: m_batch_size,
                                                self.is_training: False})
            emb_array[lab,:] = emb
            label_check_array[lab] = 1
        self.logger.info('[{}]: Forward pass: {:.3f}'.format(prefix, time.time()-start_time))
        
        assert(np.all(label_check_array==1))
        
        anchor_embeddings = emb_array[0::2]
        unknown_embeddings = emb_array[1::2]
        unknown_isequal = np.asarray(actual_issame)
        threshold_space = np.arange(0, 4, 0.01)
        tpr, fpr, accuracy, acc_thresholds = datautils.calculate_roc(
            threshold_space, 
            anchor_embeddings, 
            unknown_embeddings, 
            unknown_isequal, 
            nrof_folds=args.nrof_cross_vali_folds)

        val, val_std, far, val_thresholds = datautils.calculate_val(
            threshold_space, 
            anchor_embeddings, 
            unknown_embeddings, 
            unknown_isequal, 
            far_target=1e-3, 
            nrof_folds=args.nrof_cross_vali_folds)

        accuracy_mean = np.mean(accuracy)
        accuracy_std = np.std(accuracy)
        acc_threshold_mean = np.mean(acc_thresholds)
        acc_threshold_std = np.std(acc_thresholds)

        val_threshold_mean = np.mean(val_thresholds)
        val_threshold_std = np.std(val_thresholds)

        self.logger.info('[{}]: Accuracy: {:1.3f}±{:1.3f}'.format(prefix, accuracy_mean, accuracy_std))
        self.logger.info('[{}]: Accuracy threshold: {:2.5f}±{:2.5f}'.format(prefix, acc_threshold_mean, acc_threshold_std))
        self.logger.info('[{}]: Validation rate: {:2.5f}±{:2.5f} @ FAR={:2.5f}'.format(prefix, val, val_std, far))
        self.logger.info('[{}]: Validation threshold: {:2.5f}±{:2.5f}'.format(prefix, val_threshold_mean, val_threshold_std))
        eval_time = time.time() - start_time
        # Add validation loss and accuracy to summary
        summary = tf.Summary()

        summary.value.add(tag='{}_evaluation/accuracy'.format(prefix), simple_value=accuracy_mean)
        summary.value.add(tag='{}_evaluation/accuracy_threshold'.format(prefix), simple_value=acc_threshold_mean)
        summary.value.add(tag='{}_evaluation/val_rate'.format(prefix), simple_value=val)
        summary.value.add(tag='{}_evaluation/val_threshold'.format(prefix), simple_value=val_threshold_mean)
        summary.value.add(tag='{}_evaluation/far'.format(prefix), simple_value=far)
        summary.value.add(tag='time/{}_evaluation'.format(prefix), simple_value=eval_time)
        summary_writer.add_summary(summary, step)


    @abstractmethod
    def build_cnn(self, args, batch_images, is_training, prob_keep, kernel_regularizer, kernel_initializer, activation_fn):
        pass

    
#https://arxiv.org/pdf/1503.03832.pdf
class InceptionNN2(FaceNetModel):
    def __init__(self, model_args):
        super().__init__(model_args)
       
    def build_cnn(self, args, batch_images, is_training, kernel_regularizer, kernel_initializer, activation_fn):
        #TODO I changed a lot so I removed this impl, no longer necessary
        raise Exception('Not implemented (anymore).')

    
#https://github.com/cmusatyalab/openface
class InceptionNN4Small2(FaceNetModel):
    def __init__(self, model_args):
        super().__init__(model_args)
    

    def build_cnn(self, args, batch_images, is_training, prob_keep, kernel_regularizer, kernel_initializer, activation_fn):
        #TODO https://github.com/cmusatyalab/openface/blob/master/models/openface/nn4.small2.def.lua
        raise Exception('Not implemented.')
        


#https://arxiv.org/pdf/1602.07261.pdf
class InceptionResNetV1(FaceNetModel):
    def __init__(self, model_args):
        super().__init__(model_args)
    

    def build_cnn(self, args, batch_images, is_training, prob_keep, kernel_regularizer, kernel_initializer, activation_fn):

        conv_default = partial(model_layers.conv_layer,
                               is_training=is_training,
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=kernel_regularizer)

        conv_relu = partial(conv_default, batch_norm=True, activation=activation_fn)

        stem_out = model_layers.inception_resnet_v1_stem('stem', batch_images, conv_relu)

        # 5 x Inception-resnet-A
        block35_branch = partial(model_layers.block35, branch_conv=conv_relu, 
                                 up_conv=conv_default, out_activ_fn=activation_fn, 
                                 scale=0.17)

        inception_resnet_a = model_layers.conv_repeat('inception_resnet_a', stem_out, 5, block35_branch)
        
        # Reduction-A
        reduction_a = model_layers.reduction_a('reduction_a', inception_resnet_a, 192, 192, 256, 384, conv_relu)

        # 10 x Inception-Resnet-B
        block17_branch = partial(model_layers.block17, branch_conv=conv_relu, 
                                 up_conv=conv_default, out_activ_fn=activation_fn, 
                                 scale=0.10)

        inception_resnet_b = model_layers.conv_repeat('inception_resnet_b', reduction_a, 10, block17_branch)

        # Reduction-B
        reduction_b = model_layers.reduction_b('reduction_b', inception_resnet_b, conv_relu)

        # 5 x Inception-Resnet-C
        block8 = partial(model_layers.block8, branch_conv=conv_relu, up_conv=conv_default)

        block8_branch = partial(block8, out_activ_fn=activation_fn, scale=0.20)

        inception_resnet_c = model_layers.conv_repeat('inception_resnet_c', reduction_b, 5, block8_branch)

        block8_final = block8('block8', inception_resnet_c, out_activ_fn=None)
        
        avgpool_1a_8x8 = tf.layers.average_pooling2d(block8_final, block8_final.get_shape()[1:3], 
                                                     strides=1, padding='VALID', name='avgpool_1a_8x8')

        return avgpool_1a_8x8

    
#https://arxiv.org/pdf/1602.07261.pdf
class InceptionResNetV1Small(FaceNetModel):
    def __init__(self, model_args):
        super().__init__(model_args)
    

    def build_cnn(self, args, batch_images, is_training, prob_keep, kernel_regularizer, kernel_initializer, activation_fn):

        conv_default = partial(model_layers.conv_layer,
                               is_training=is_training,
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=kernel_regularizer)

        conv_relu = partial(conv_default, batch_norm=True, activation=activation_fn)

        stem_out = model_layers.inception_resnet_v1_stem('stem', batch_images, conv_relu)

        # 5 x Inception-resnet-A
        block35_branch = partial(model_layers.block35, branch_conv=conv_relu, 
                                 up_conv=conv_default, out_activ_fn=activation_fn, 
                                 scale=0.17)

        inception_resnet_a = model_layers.conv_repeat('inception_resnet_a', stem_out, 5, block35_branch)
        
        # Reduction-A
        reduction_a = model_layers.reduction_a('reduction_a', inception_resnet_a, 192, 192, 256, 384, conv_relu)

        # 10 x Inception-Resnet-B
        block17_branch = partial(model_layers.block17, branch_conv=conv_relu, 
                                 up_conv=conv_default, out_activ_fn=activation_fn, 
                                 scale=0.10)

        inception_resnet_b = model_layers.conv_repeat('inception_resnet_b', reduction_a, 10, block17_branch)

        # Reduction-B
        reduction_b = model_layers.reduction_b('reduction_b', inception_resnet_b, conv_relu)

        # 5 x Inception-Resnet-C
        block8 = partial(model_layers.block8, branch_conv=conv_relu, up_conv=conv_default)

        block8_branch = partial(block8, out_activ_fn=activation_fn, scale=0.20)

        inception_resnet_c = model_layers.conv_repeat('inception_resnet_c', reduction_b, 5, block8_branch)

        block8_final = block8('block8', inception_resnet_c, out_activ_fn=None)
        
        avgpool_1a_8x8 = tf.layers.average_pooling2d(block8_final, block8_final.get_shape()[1:3], 
                                                     strides=1, padding='VALID', name='avgpool_1a_8x8')

        return avgpool_1a_8x8