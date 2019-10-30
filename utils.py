import os
import numpy as np
import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape,Activation
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Deconvolution2D,Conv3D,MaxPooling3D
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
from numpy.random import random,choice
import tensorflow as tf
import logging

def normalize_images(images):
    images = images.astype('float32')
    images = (images -127.5)/127.5
    return images

def denormalize_images(images):
    images = (images * 127.5) + 127.5;
    return images

def smooth_positive_labels(y):
    return y - 0.3 + (random(y.shape) * 0.5)

def smooth_negativ_labels(y):
    return y + (random(y.shape) * 0.3)

def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * y.shape[0])
    # choose labels to flip
    flip_ix = choice([i for i in range(y.shape[0])], size=n_select)
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix] 
    return y

def get_fake_labels(num):
    labels = np.zeros(num)
    labels = smooth_negativ_labels(labels)
    return noisy_labels(labels,0.05)
    
def get_real_labels(num):
    labels = np.ones(num)
    labels = smooth_positive_labels(labels)
    return noisy_labels(labels,0.05)

def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)
