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

import random
import threading
import numpy as np
import pickle


class Identity(object):
    def __init__(self, id, name, color):
        self.id = id
        self.name = name
        self.color = color
        self.embeddings = np.array([], dtype=float)
        self.avg_embedding = np.array([], dtype=float)

    def __repr__(self):
        return 'id={}, name={}, color={}, nrof_embs={}'.format(self.id, self.name, self.color, self.embeddings.shape[0])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.embeddings.shape[0]

    def serialize(self):
        return dict(
            id=self.id,
            name=self.name,
            size=len(self),
            color=self.color)


class IdentityStore(object):
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            #pickle.HIGHEST_PROTOCOL
            pickle.dump(self, f, -1)

    def __init__(self, face_emb_size):
        self.face_emb_size = face_emb_size
        self._identities = []
        self._top_id = -1
        self._ident_colors = set()
        self._lock = threading.Lock()

    def get_next_id(self):
        with self._lock:
            return self._top_id + 1

    def add_identity(self, name, initial_embedding):
        assert initial_embedding.shape[0] == self.face_emb_size
        return self.add_identity_with_embeddings(name, np.expand_dims(initial_embedding, axis=0))
    
    def add_identity_with_embeddings(self, name, embeddings):
        assert embeddings.shape[1] == self.face_emb_size
        with self._lock:
            new_ident = self._create_new_identity(name)
            new_ident.embeddings = np.array(embeddings)
            new_ident.avg_embedding = np.average(new_ident.embeddings, axis=0)
            return new_ident
        
    def _create_new_identity(self, name):
        self._top_id += 1
        next_color = generate_new_color(self._ident_colors, pastel_factor=0.8)
        self._ident_colors.add(next_color)
        new_ident = Identity(self._top_id, name, next_color)
        self._identities.append(new_ident)
        return new_ident

    def add_embedding(self, id, embedding):
        assert embedding.shape[0] == self.face_emb_size
        self.add_embeddings(id, np.expand_dims(embedding, axis=0))

    def add_embeddings(self, id, embeddings):
        assert embeddings.shape[1] == self.face_emb_size
        with self._lock:
            ident = self._identities[id]
            ident.embeddings = np.append(ident.embeddings, np.array(embeddings), axis=0)
            ident.avg_embedding = np.average(ident.embeddings, axis=0)
        
    def find_identity(self, embedding, threshold):
        assert embedding.shape[0] == self.face_emb_size
        with self._lock:
            min_dist = float('inf')
            min_ident = None
            for ident in self._identities:
                ident_dist = np.square(np.linalg.norm(ident.avg_embedding - embedding))
                if ident_dist < min_dist:
                    min_dist = ident_dist
                    min_ident = ident
            if min_dist < threshold:
                return min_ident
            else:
                return None

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_lock']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._lock = threading.Lock()
        
    def __len__(self):
        return len(self._identities)

    @property
    def identities(self):
        return self._identities


def get_random_color(pastel_factor = 0.5):
    return tuple(int(255 * ((x + pastel_factor)/(1.0 + pastel_factor))) for x in [random.uniform(0, 1.0) for i in [1, 2, 3]])


def color_distance(c1, c2):
    return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])


def generate_new_color(existing_colors, pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color, c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color
