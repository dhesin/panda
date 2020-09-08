from tensorflow_examples.models.pix2pix import pix2pix
import openslide
import skimage.io
import pandas as pd
import PIL
import cv2
import io
import os

# General packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display
from IPython.display import clear_output
import fnmatch

import tensorflow as tf

image_mask_feature_description = {
    'image': tf.io.VarLenFeature(tf.int64),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'provider': tf.io.VarLenFeature(tf.string),
    'mask': tf.io.VarLenFeature(tf.int64),
    'gleason': tf.io.VarLenFeature(tf.string),
}

def filter_tfrecord_2(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    #print(example_proto)
    image_features = tf.io.parse_example(example_proto, image_mask_feature_description)

    num_blocks = image_features['depth']
    if (num_blocks>1500):
        return False
    return True



listOfFiles = os.listdir("/tf/kaggle/tf_records")
pattern = "*.tfrecords"
listOfFiles = ["/tf/kaggle/tf_records/"+x for x in listOfFiles if fnmatch.fnmatch(x,pattern)]
print(listOfFiles)

dataset = tf.data.TFRecordDataset(listOfFiles)
print(dataset)
dataset = dataset.filter(filter_tfrecord_2)
print(dataset)
i=0
for data in dataset:
    print(i)
    i=i+1
