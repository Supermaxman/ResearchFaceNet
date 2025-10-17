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

import os
import cv2
import time
import pickle
import numpy as np
import threading
import glob

import align_dlib
import face_identity
import face_embedding
import datautils

if __name__ == '__main__':
    face_img_size = 96
    face_emb_size = 128
    #input_dir = 'IdentityStore'
    #output_path = 'identity-store-old.pkl'
    #input_dir = 'D:/Data/lfw'
    input_dir = 'IdentityStoreLFW'
    output_path = 'identity-store.pkl'
    model_path = 'InceptionResNetV1Small-VGGFace2-v1-1000.pb'
        
    if os.path.isfile(output_path):
        i_store = face_identity.IdentityStore.load(output_path)
        print('Loaded {} identities.'.format(len(i_store)))
    else:
        
        dlib_model = align_dlib.AlignDlib('shape_predictor_68_face_landmarks.dat')

        e_model = face_embedding.EmbeddingModel(face_img_size, model_path)
        e_model.load()

        i_store = face_identity.IdentityStore(face_emb_size)
    
        print('Processing identity files...')
        for ident_dir in os.listdir(input_dir):
            ident_name = os.path.basename(ident_dir)
            ident_input_dir = os.path.join(input_dir, ident_name)
            ident_image_paths = glob.glob(os.path.join(ident_input_dir, '*.jpg'))
            ident_images = []
            print('Processing {} images for {} in {}'.format(len(ident_image_paths), ident_name, ident_input_dir))
            for ident_image_path in ident_image_paths:
                ident_image = datautils.process_image(ident_image_path, face_img_size, dlib_model)
                if ident_image is not None:
                    ident_images.append(ident_image)
            if len(ident_images) > 0:
                ident_embeddings = e_model.create_embeddings(ident_images)
                i_store.add_identity_with_embeddings(ident_name, ident_embeddings)

        print('Saving IdentityStore...')
        i_store.save(output_path)
        print('Done')


