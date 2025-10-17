# MIT License
# 
# Copyright (c) 2016 David Sandberg
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

# some functions from https://github.com/davidsandberg/facenet and modified for slightly different usage

import logging
import argparse
import os
import time
import multiprocessing as mp

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import data_flow_ops
import cv2
from PIL import Image
import glob

from sklearn.model_selection import KFold
from scipy import interpolate
import itertools
import random

import align_dlib

logger = logging.getLogger(__name__)

def load_graph(graph_filename, input_map=None, prev_graph=None):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    if prev_graph is None:
        g = tf.Graph()
    else:
        g = prev_graph
    #TODO removing graph shape info
    with g.as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=input_map, 
            return_elements=None, 
            name=''
        )
    return graph


class ClassData():
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '{}: {} images'.format(self.name, len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

def get_dataset(data_directory):
    dataset = []
    image_count = 0
    class_folders = os.listdir(data_directory)
    class_folders.sort()
    for class_name in class_folders:
        class_directory = os.path.join(data_directory, class_name)
        if os.path.isdir(class_directory):
            image_paths = [os.path.join(class_directory, img) for img in os.listdir(class_directory)]
            image_paths.sort()
            dataset.append(ClassData(class_name, image_paths))
            image_count += len(image_paths)
        else:
            logger.info('Skipping non-directory file: {}'.format(class_directory))
    return dataset, image_count

def filter_dataset(dataset, min_images_per_label=6):
    filtered_dataset = []
    image_count = 0
    for class_data in dataset:
        class_data_length = len(class_data)
        if class_data_length < min_images_per_label:
            logger.info('Skipping class: {} - not enough images ({})'.format(class_data.name, class_data_length))
        else:
            filtered_dataset.append(class_data)
            image_count += class_data_length
    return filtered_dataset, image_count

def split_dataset(dataset, split_ratio=0.8, min_num_images=2, shuffle=True):
    train_set = []
    train_image_count = 0
    test_set = []
    test_image_count = 0
    for class_data in dataset:
        paths = class_data.image_paths
        if shuffle:
            np.random.shuffle(paths)
        split = int(np.floor(len(paths) * split_ratio))
        if split < min_num_images:
            logger.info('Skipping class: {} - not enough images for train/test split ({})'.format(class_data.name, split))
            continue
        train_clsd = ClassData(class_data.name, paths[:split])
        test_clsd = ClassData(class_data.name, paths[split:])
        train_image_count += len(train_clsd)
        test_image_count += len(test_clsd)
        train_set.append(train_clsd)
        test_set.append(test_clsd)
    return train_set, test_set, train_image_count, test_image_count


def preprocess_image(args):
    """
    Detect face, align and crop :param input_path. Write output to :param output_path
    :param input_path: Path to input image
    :param output_path: Path to write processed image
    :param crop_dim: dimensions to crop image to
    """
    input_path, output_path, crop_dim, dlib_model = args
    image = process_image(input_path, crop_dim, dlib_model)
    if image is not None:
        # BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        logger.debug('Writing processed file: {}'.format(output_path))
        cv2.imwrite(output_path, image)
    else:
        logger.warning('Skipping filename: {}'.format(input_path))
        
def process_image(filename, crop_dim, dlib_model):
    image = None
    aligned_image = None

    image = _buffer_image(filename)

    if image is not None:
        aligned_image = _align_image(image, crop_dim, dlib_model)
    else:
        raise IOError('Error buffering image: {}'.format(filename))

    return aligned_image


def _buffer_image(filename):
    logger.debug('Reading image: {}'.format(filename))
    # BGR
    image = cv2.imread(filename, )
    # RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def _align_image(image, crop_dim, dlib_model):
    # RGB
    bb = dlib_model.getLargestFaceBoundingBox(image)
    #AlignDlib.INNER_EYES_AND_BOTTOM_LIP vs AlignDlib.OUTER_EYES_AND_NOSE ? 
    # RGB
    aligned = dlib_model.align(crop_dim, image, bb, landmarkIndices=align_dlib.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    return aligned


def run_preprocess(input_dir, output_dir, crop_dim, nrof_threads):
    start_time = time.time()
    pool = mp.Pool(processes=nrof_threads)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_dir in os.listdir(input_dir):
        image_output_dir = os.path.join(output_dir, os.path.basename(os.path.basename(image_dir)))
        image_input_dir = os.path.join(input_dir, os.path.basename(os.path.basename(image_dir)))
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)
            
        image_paths = glob.glob(os.path.join(image_input_dir, '*.jpg'))
        dlib_model = align_dlib.AlignDlib('shape_predictor_68_face_landmarks.dat')
        #TODO look for better way to share dlib_model or create one per process.
        pool.map(preprocess_image, [(image_path, os.path.join(image_output_dir, os.path.basename(image_path)), crop_dim, dlib_model) for image_path in image_paths])
        
    
    #image_folders = glob.glob(os.path.join(input_dir, '*'))

    #image_paths = glob.glob(os.path.join(input_dir, '**/*.jpg'))
    #for index, image_path in enumerate(image_paths):
    #    image_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
    #    output_path = os.path.join(image_output_dir, os.path.basename(image_path))
    #    pool.apply_async(preprocess_image, (image_path, output_path, crop_dim))

    pool.close()
    pool.join()
    logger.info('Completed in {} seconds'.format(time.time() - start_time))

def read_data(image_paths, labels, 
              image_size, batch_size, 
              num_threads, queue_capacity,
              shuffle, random_flip, random_brightness, random_contrast, random_crop):


    input_queue = tf.FIFOQueue(capacity=queue_capacity,
                                dtypes=[tf.string, tf.int64],
                                shapes=[[3], [3]], shared_name=None, name=None)
    enqueue_op = input_queue.enqueue_many([image_paths, labels], name='enqueue_op')

    images_and_labels = []
    for _ in range(num_threads):
        filenames, label = input_queue.dequeue()
        images = []
        for filename in tf.unstack(filenames):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_image(file_contents, channels=3)

            if random_crop:
                image = tf.random_crop(image, [image_size, image_size, 3])
            else:
                image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)

            #TODO add more rotational invariance (tilt)
            if random_flip:
                image = tf.image.random_flip_left_right(image)

            if random_brightness:
                image = tf.image.random_brightness(image, max_delta=0.3)

            if random_contrast:
                image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            
            
            image.set_shape((image_size, image_size, 3))
            image = tf.image.per_image_standardization(image)
            images.append(image)
        images_and_labels.append([images, label])
    
    image_batch, label_batch = tf.train.batch_join(
        images_and_labels, batch_size=batch_size, 
        shapes=[[image_size, image_size, 3], []], 
        enqueue_many=True,
        capacity=4 * num_threads,
        allow_smaller_final_batch=True)

    image_batch = tf.identity(image_batch, 'image_batch')
    image_batch = tf.identity(image_batch, 'input')
    label_batch = tf.identity(label_batch, 'label_batch')

    return image_batch, label_batch, enqueue_op



def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for index, class_data in enumerate(dataset):
        image_paths_flat += class_data.image_paths
        labels_flat += [index] * len(class_data)
    return image_paths_flat, labels_flat


def get_train_and_test_set(input_dir, min_num_images_per_label=10, split_ratio=0.8):
    
    dataset, total_image_count = get_dataset(input_dir)
    dataset, filtered_image_count = filter_dataset(dataset, min_images_per_label=min_num_images_per_label)
    train_set, test_set, train_image_count, test_image_count = split_dataset(dataset, split_ratio=split_ratio)

    return train_set, test_set, train_image_count, test_image_count


def load_images_and_labels(dataset, image_size, batch_size, num_threads, queue_capacity, num_epochs, random_flip=False, random_brightness=False, random_contrast=False):
    class_names = [cls.name for cls in dataset]
    image_paths, labels = get_image_paths_and_labels(dataset)
    images, labels = read_data(image_paths, labels, image_size, batch_size, num_epochs, num_threads, queue_capacity, 
                               shuffle=False, 
                               random_flip=random_flip, 
                               random_brightness=random_brightness,
                               random_contrast=random_contrast)
    return images, labels, class_names


def sample_people(dataset, people_per_batch, images_per_person):
    # calculate the number of images we will consider for this sample
    nrof_images = people_per_batch * images_per_person
  
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    next_class_idx = 0
    image_paths = []
    num_per_class = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        # get next random class index
        # this will throw an index exception if we do not have enough data to sample nrof_images from,
        # or if all those images are within a very small number of classes.
        class_index = class_indices[next_class_idx]
        # get number of images for specified class
        nrof_images_in_class = len(dataset[class_index])
        # ramdomly sample images by shuffling them before indexing.
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        # best case senario we have images_per_person as a minimum, which means we get a full sample from this class
        # second best case is we take all the images in the class, since they are less than the optimal images_per_person we want.
        # third case occurs when we want to avoid oversampling the maximum nrof_images, and so we pick the remaining number to finish the loop.
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        # sample from random images for this class based on the best case number of images.
        idx = image_indices[0:nrof_images_from_class]
        # get the image paths for these indexes
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        # add image paths to list
        image_paths += image_paths_for_class
        # keep track of the number of images we sampled from this class (optimally images_per_person)
        num_per_class.append(nrof_images_from_class)
        # incriment to move to the next class
        next_class_idx += 1

    return image_paths, num_per_class

def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha, 
                    semi_hard_negatives=False, pairwise_triplet_negatives=False):
    """ Select the triplets for training
    """
    emb_start_idx = 0
    nrof_possible_triplets = 0
    triplets = []

    # iterate through the sampled image classes
    for id_idx in range(people_per_batch):
        # get the number of images for the sampled class
        nrof_images = int(nrof_images_per_class[id_idx])
        # a_idx iterates from first to second-to-last image/embedding indexes
        # p_idx iterates from a_idx + 1 to last image/embedding indexes
        # this creates all combination pairs of a_idx and p_idx, and then for each 
        # combination we sample an n_idx creating (a_idx, p_idx, n_idx) triplets. 

        if pairwise_triplet_negatives:
            # calculate distance from every embedding for the class to every embedding 
            # [cls_emb_idx, emb_idx]
            # just do this once and store it for use during loop.
            idx_dists = np.sum(np.square(embeddings[emb_start_idx:emb_start_idx + nrof_images, None] - embeddings[None, :]), axis=2)

        for a_idx in range(emb_start_idx, emb_start_idx + nrof_images - 1):
            # since we are going to need all positive distances we might as well calculate them all at once
            if pairwise_triplet_negatives:
                neg_dists_sqr = idx_dists[a_idx - emb_start_idx]
            else:
                # calculate distances from anchor embedding to all other embeddings
                neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), axis=1)

            # For every possible positive pair.
            for p_idx in range(a_idx + 1, emb_start_idx + nrof_images): 
                # calculate the distance from the anchor embedding to the positive embedding
                if pairwise_triplet_negatives:
                    pos_dist_sqr = idx_dists[a_idx - emb_start_idx, p_idx - emb_start_idx]
                else:
                    pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                
                # check whether the margin has been violated
                selection_criteria = pos_dist_sqr + alpha > neg_dists_sqr

                # unsure whether semi-hard or just violating hard negatives are better.
                # seems like semi-hard may be more difficult to find convergence within a reasonable timeframe.
                # consider using semi-hard after mostly trained?
                if semi_hard_negatives:
                    # FaceNet selection of semi-hard negatives, 
                    # triplets that violate the margin but are still farther than positives.
                    # these are the triplets that lie within the margin, and represent the hardest triplets.
                    selection_criteria = np.logical_and(selection_criteria, pos_dist_sqr < neg_dists_sqr) 
                # See comments in "face_models.py"
                if pairwise_triplet_negatives:
                    pos_neg_dists_sqr = idx_dists[p_idx - emb_start_idx]
                    #TODO consider logical_or?
                    selection_criteria = np.logical_and(selection_criteria, pos_dist_sqr + alpha> pos_neg_dists_sqr) 
                    if semi_hard_negatives:
                        selection_criteria = np.logical_and(selection_criteria, pos_dist_sqr < pos_neg_dists_sqr) 

                # ignore images from the same class
                selection_criteria[emb_start_idx:emb_start_idx + nrof_images] = False

                all_neg = np.where(selection_criteria)[0] 

                # if any negative images found then randomly sample one and create triplet
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))

                nrof_possible_triplets += 1
        # when done with a class increment start index by number of images in class
        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, nrof_possible_triplets, len(triplets)


def calculate_roc(thresholds, anchor_embeddings, unknown_embeddings, unknown_isequal, nrof_folds=10):
    assert(anchor_embeddings.shape[0] == unknown_embeddings.shape[0])
    assert(anchor_embeddings.shape[1] == unknown_embeddings.shape[1])
    nrof_pairs = min(len(unknown_isequal), anchor_embeddings.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    acc_thresholds = np.zeros((nrof_folds))
    
    diff = np.subtract(anchor_embeddings, unknown_embeddings)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], unknown_isequal[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], unknown_isequal[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], unknown_isequal[test_set])
        acc_thresholds[fold_idx] = thresholds[best_threshold_index]
          
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, acc_thresholds

def calculate_accuracy(threshold, dist, actual_issame):
    # predicted same if less than threshold
    predict_issame = np.less(dist, threshold)

    # true positives
    tp = np.sum(np.logical_and(predict_issame, actual_issame))

    # false positives
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    # true negatives
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))

    # false negatives
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    
    # true positive rate
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)

    # false positive rate
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)

    # accuracy
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


def calculate_val_far(threshold, dist, actual_issame):
    # predicted same if less than threshold
    predict_issame = np.less(dist, threshold)

    # true positives
    tp = np.sum(np.logical_and(predict_issame, actual_issame))

    # false positives
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    # number same
    n_same = np.sum(actual_issame)

    # number different
    n_diff = np.sum(np.logical_not(actual_issame))

    # val is ratio of true positives to total same
    val = float(tp) / float(n_same)

    # false accept rate is ratio of false positives to total different
    far = float(fp) / float(n_diff)

    return val, far


def calculate_val(thresholds, anchor_embeddings, unknown_embeddings, unknown_isequal, far_target, nrof_folds=10):
    assert(anchor_embeddings.shape[0] == unknown_embeddings.shape[0])
    assert(anchor_embeddings.shape[1] == unknown_embeddings.shape[1])
    nrof_pairs = min(len(unknown_isequal), anchor_embeddings.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    val_thresholds = np.zeros(nrof_folds)

    diff = np.subtract(anchor_embeddings, unknown_embeddings)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        # get the false accept rates for all thresholds
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], unknown_isequal[train_set])
        # if the max false accept rate is greater than or equal to the false accept rate target
        # then we need to interpolate a threshold that will give us a closer false accept rate
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        # if the max false accept rate is less than the far target then #TODO not sure what to do here...
        else:
            threshold = 0.0
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], unknown_isequal[test_set])
        val_thresholds[fold_idx] = threshold
    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean, val_thresholds


def generate_pairs(data, nrof_sample_classes, nrof_anchors, nrof_positive_pairs_per_anchor, nrof_negative_pairs_per_anchor):
    pairs = []
    for sample_idx in random.sample(range(len(data)), nrof_sample_classes):
        sample_data = data[sample_idx]
        for anchor_idx in random.sample(range(len(sample_data)), min(nrof_negative_pairs_per_anchor, len(sample_data))):
            anchor = sample_data.image_paths[anchor_idx]
            positive_count = 0
            for positive_idx in random.sample([x for x in range(len(sample_data)) if x != anchor_idx], min(nrof_positive_pairs_per_anchor, len(sample_data) - 1)):
                positive = sample_data.image_paths[positive_idx]
                pairs.append((anchor, positive, True))
                positive_count += 1

            negative_count = 0
            while negative_count < nrof_negative_pairs_per_anchor and negative_count < positive_count:
                for negative_class_idx in random.sample([x for x in range(len(data)) if x != sample_idx], 1):
                    negative_data = data[negative_class_idx]
                    if len(negative_data) < 1:
                        continue
                    for negative_idx in random.sample(range(len(negative_data)), 1):
                        negative = negative_data.image_paths[negative_idx]
                        pairs.append((anchor, negative, False))
                        negative_count += 1

    random.shuffle(pairs)
    # needs to be divisable by three due to model image queue setup
    pairs = pairs[:len(pairs) - (len(pairs) % 3)]
    return pairs

def flatten_pairs(pairs):
    flat_pairs = []
    actual_issame = []

    for anchor, unknown, issame in pairs:
        flat_pairs.append(anchor)
        flat_pairs.append(unknown)
        actual_issame.append(issame)

    return flat_pairs, actual_issame


def write_pairs(pairs, folder_prefix, filepath):
    with open(filepath, 'x') as file:
        for anchor, unknown, issame in pairs:
            file.write('{},{},{}\n'.format(anchor[len(folder_prefix):], unknown[len(folder_prefix):], int(issame)))


def get_lfw_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.jpg')
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.jpg')
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.jpg')
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.jpg')
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        logger.info('Skipped {} image pairs'.format(nrof_skipped_pairs))
    
    return path_list, issame_list

def read_lfw_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def read_pairs(filepath, folder_prefix):
    pairs = []
    with open(filepath, 'r') as file:
        for line in file:
            sp = line.split(',')
            anchor = folder_prefix + sp[0]
            unknown = folder_prefix + sp[1]
            issame = bool(int(sp[2]))
            pairs.append((anchor, unknown, issame))
    return pairs


def load_pairs(pairs_file, input_directory, data_set, nrof_anchors, nrof_positive_pairs_per_anchor, nrof_negative_pairs_per_anchor):
    if not os.path.isfile(pairs_file):
        logger.info('No pairs found, generating...')
        pairs = generate_pairs(data_set, len(data_set), nrof_anchors, nrof_positive_pairs_per_anchor, nrof_negative_pairs_per_anchor)
        # needs to be divisable by three due to model image queue setup
        pairs = pairs[:len(pairs) - ((2 * len(pairs)) % 3)]
        write_pairs(pairs, input_directory, pairs_file)
    else:
        pairs = read_pairs(pairs_file, input_directory)
    return pairs



# TODO move this into a run_preprocess.py
if __name__ == '__main__':    
    SEED=0
    np.random.seed(SEED)
    random.seed(SEED)
    logging.basicConfig(level=logging.INFO)
    img_size=96
    threads=6
    run_preprocess('D:/Data/vgg_face2_train', 'D:/Data/vgg_face2_train_p_{}x{}'.format(img_size, img_size), img_size, threads)
    run_preprocess('D:/Data/lfw', 'D:/Data/lfw_p_{}x{}'.format(img_size, img_size), img_size, threads)





