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
import tf_utils
import numpy as np
        
class EmbeddingModel(object):
    def __init__(self, face_img_size, graph_file):
        self.face_img_size = face_img_size
        self.graph_file = graph_file

    def load(self):
        # set up image standardization as part of the image feed graph
        # this lets us avoid using any of the fifo queue built for image paths and not
        # image matrices
        with tf.Graph().as_default() as p_graph:
            
            self.image_batch_p = tf.placeholder(
                dtype=tf.float32, 
                shape=[None, self.face_img_size, self.face_img_size, 3], 
                name='image_batch_p')
            
            self.image_batch_s = tf.map_fn(
                lambda frame: tf.image.per_image_standardization(frame), 
                self.image_batch_p, 
                name='image_batch_s')

            # also might as well set these once and avoid passing them in every sess.run.
            is_training_p = tf.constant(False, dtype=tf.bool, name='is_training_p')
            prob_keep_p = tf.constant(1.0, dtype=tf.float32, name='prob_keep_p')
        #TODO issue: this is removing tensor shape information
        self.graph = tf_utils.load_graph(
            self.graph_file, 
            input_map={
                'inputs/image_batch:0': self.image_batch_s,
                'inputs/is_training:0': is_training_p,
                'inputs/prob_keep:0': prob_keep_p},
            prev_graph=p_graph)
        
        self.image_batch = self.graph.get_tensor_by_name('image_batch_p:0')
        self.embeddings = self.graph.get_tensor_by_name('outputs/embeddings:0')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

    def create_embeddings(self, image_batch):
        return self.sess.run(
            self.embeddings, 
            feed_dict={self.image_batch: image_batch})

    def create_embedding(self, image):
        return np.squeeze(
            self.create_embeddings(
                np.expand_dims(image, axis=0)), 
            axis=0)

    def get_activations(self, layer, image):
        return np.squeeze(
            self.sess.run(
                layer, 
                feed_dict={self.image_batch: np.expand_dims(image, axis=0)}), 
            axis=0)

    def get_tensor(self, name):
        return self.graph.get_tensor_by_name(name)

    def get_operation(self, name):
        return self.graph.get_operation_by_name(name)