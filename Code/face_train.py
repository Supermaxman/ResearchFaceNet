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
import argparse
import os
import time
import json
import numpy as np
import tensorflow as tf
import random

import face_models


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    SEED=555
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    random.seed(SEED)
    start_time = time.time()
    
    model_name='InceptionResNetV1Small-VGGFace2-v1'
    load_model = True

    model_directory = os.path.join('D:\\Models\\ResearchFaceNet\\models', model_name)
    
    with open(os.path.join(model_directory, model_name + '.model'),'rt') as file:
        args = json.loads(file.read())
    
    model_args = type('ModelArgs', (object,), args['model'])
    model_args.model_name=model_name
    model_args.model_directory=model_directory

    train_args = type('TrainingArgs', (object,), args['train'])
    train_args.log_path = os.path.join(train_args.log_directory, model_name)

    if model_args.type == 'InceptionResNetV1Small':
        model = face_models.InceptionResNetV1Small(model_args)
    elif model_args.type == 'InceptionResNetV1':
        model = face_models.InceptionResNetV1(model_args)
    elif model_args.type == 'InceptionNN2':
        model = face_models.InceptionNN2(model_args)
    else:
        raise Exception('Unknown model specified!')
    

    #if load_model:
    #    model.load(trainable=True)
    #else:
    #    model.create()
    model.freeze()
    #model.train(train_args)
