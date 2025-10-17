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

import numpy as np
import random
import cProfile
import time

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import face_identity
import face_embedding

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def generate_random_embedding(embedding_size):
    rand_vec = np.random.uniform(size=[embedding_size])
    rand_norm_vec = normalized(rand_vec)[0]
    rand_norm = np.linalg.norm(rand_norm_vec)
    return rand_norm_vec

#Python Lists:
#Took 0.146530 seconds to run 1000 tests.

#Numpy Array per Identity & dict of identities
#Took 0.074001 seconds to run 1000 tests.

#Numpy Array per Identity & list of identities
#Took 0.072003 seconds to run 1000 tests.

#Numpy Array for all embeddings & embedding to identity Numpy Array
#TODO

#TODO ROC curve for model & for IdentityStore  

def speed_test():
    face_img_size = 96
    embedding_size = 128
    threshold = 0.925
    num_tests = 1000
    ident_store = face_identity.IdentityStore.load('identity-store.pkl')
    elapsed_time = 0.0
    for test_num in range(num_tests):
        test_start_time = time.time()
        test_emb = generate_random_embedding(embedding_size)
        found_ident = ident_store.find_identity(test_emb, threshold)
        elapsed_time += time.time() - test_start_time
    print('Took {:.6f} seconds to run {} tests.'.format(elapsed_time, num_tests))

def emb_test():
    embedding_size = 128
    ident_store = face_identity.IdentityStore.load('identity-store.pkl')
    ident_length = len(ident_store)
    X = None
    Y = None
    C = None
    S = None
    default_shape = 10
    avg_shape = 10

    for idx, ident in enumerate(ident_store._identities):
        ident_emb_array = ident.embeddings
        ident_emb_array = np.append(ident_emb_array, np.expand_dims(ident.avg_embedding, axis=0), axis=0)

        ident_emb_length = ident_emb_array.shape[0]
        ident_id_array = np.array([ident.id] * ident_emb_length, dtype=int)
        ident_color_array = np.array([ident.color] * ident_emb_length, dtype=float) / 255
        ident_color_array[-1] = ident_color_array[-1] / 2
        ident_shape_array = np.array([default_shape] * ident_emb_length, dtype=int)
        ident_shape_array[-1] = avg_shape
        if X is None:
            X = ident_emb_array
            Y = ident_id_array
            C = ident_color_array
            S = ident_shape_array
        else:
            X = np.append(X, ident_emb_array, axis=0)
            Y = np.append(Y, ident_id_array, axis=0)
            C = np.append(C, ident_color_array, axis=0)
            S = np.append(S, ident_shape_array, axis=0)
    
    tsne = TSNE(n_components=2, metric='euclidean', verbose=1)
    X_tsne = tsne.fit_transform(X)
    ##TODO color based on ident colorings
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], s=S, facecolors=C)

    prev_id = None
    prev_point = None
    eps = 0.2
    rad = 1.0
    for id, x, y in zip(Y, X_tsne[:, 0], X_tsne[:, 1]):
        if prev_id is not None and prev_id != id:
            plt.annotate(ident_store._identities[prev_id].name, (prev_point[0]+eps, prev_point[1]+eps))
            circle = plt.Circle(prev_point, rad, color='r', fill=False, linestyle='--')
            ax.add_artist(circle)
        prev_point = (x, y)
        prev_id = id
        
    plt.annotate(ident_store._identities[prev_id].name, (prev_point[0]+eps, prev_point[1]+eps))
    circle = plt.Circle(prev_point, rad, color='r', fill=False, linestyle='--')
    ax.add_artist(circle)


    plt.show()



if __name__ == '__main__':
    SEED=555
    np.random.seed(SEED)
    random.seed(SEED)
    emb_test()



    #emb_model = face_embedding.EmbeddingModel(face_img_size, 'InceptionResNetV1Small-VGGFace2-v1-1000.pb')
    #emb_model.load()
    
    #test_emb = emb_model.create_embeddings(np.zeros([1, face_img_size, face_img_size, 3]))[0]
    

